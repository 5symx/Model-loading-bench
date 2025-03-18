#include <fractos/common/experiment.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "./util.hpp"
// #include "./util.hpp"


static constexpr size_t max_in_flight = 256;


static
void
do_server(auto pch, size_t thread_idx, size_t max_threads)
{
    using namespace fractos::core;
    using namespace std::chrono;

    auto ch = pch->get_process()->make_channel(
        pch->get_config(), "srv-"+std::to_string(thread_idx)+"/"+std::to_string(max_threads))
        .get();
    auto gns = gns::make_service();

    std::array<cap::request, max_in_flight> req_server;
    std::array<gns::guard,   max_in_flight> gns_server;
    std::array<cap::request, max_in_flight> resp_client;

    for (size_t idx = 0; idx < max_in_flight; idx++) {
        req_server[idx] = ch->make_request_builder(
            ch->get_default_endpoint(),
            [&resp_client, idx](auto ch, auto args) {
                args.reset();
                ch->invoke(resp_client[idx])
                    .as_callback();
            })
            .on_channel()
            .make_request()
            .get();

        auto name = "req_server_" + std::to_string(thread_idx) + "_" + std::to_string(idx);
        gns_server[idx] = gns->publish_named(ch, req_server[idx], name)
            .get();
    }

    for (size_t idx = 0; idx < max_in_flight; idx++) {
        auto name = "resp_client_" + std::to_string(thread_idx) + "_" + std::to_string(idx);
        resp_client[idx] = gns->get_wait_for<cap::request>(ch, name, 2min)
            .get();
    }

    ready_info::send_ready(gns, pch, max_threads);
    ch->run_until([]() { return false; });
}

int main(int argc, char *argv[])
{
    using namespace fractos::common;
    using namespace fractos::core;
    using namespace std::chrono;

    logging::init(argv[0]);

    auto odesc = options();
    odesc.add_options()
        ("client", cmdline::po::bool_switch(),
         "client mode operation")
        ("server", cmdline::po::bool_switch(),
         "server mode operation")
        ;
    auto [args, pch, output, metric, control_thread, measurement_threads] = parse(odesc, argc, argv);
    signal::init_log_handler(SIGUSR1, pch->get_process());
    pch->get_process()->log_state();

    auto num_measurement_threads = measurement_threads->size();
    auto is_client = args["client"].as<bool>();
    auto is_server = args["server"].as<bool>();
    CHECK(is_client or is_server);
    if (is_client and is_server) {
        num_measurement_threads /= 2;
    }

    LOG(INFO) << "===================== DONE ==================";


    std::unique_ptr<thread::group> server_threads;
    if (is_server) {
        server_threads = thread::with_group(
            num_measurement_threads,
            [](size_t thread_idx) {
                std::string name = "srv-" + std::to_string(thread_idx);
                CHECK(pthread_setname_np(pthread_self(), name.c_str()) == 0);
                return true;
            },
            [&](size_t thread_idx, bool) {
                cpu::pin(measurement_threads->pop_front());
                do_server(pch, thread_idx, num_measurement_threads);
            });
    }

    if (not is_client) {
        server_threads->join_all();
        return 0;
    } else if (is_server) {
        server_threads->detach_all();
    }


    control_thread->pin();

    auto gns = fractos::core::gns::make_service();
    auto ready = std::make_shared<ready_info>(gns, pch);


    struct thread_state {
        std::unordered_set<size_t> available_idx;
        std::array<experiment::time_point_type, max_in_flight> start_time;
        std::function<void(experiment::time_point_type)> finish_experiment_cb;
        std::shared_ptr<channel> ch;
        std::array<cap::request, max_in_flight> req_server;
        std::array<cap::request, max_in_flight> resp_client;
        std::array<fractos::core::gns::guard, max_in_flight> gns_resp_client;
    };

    auto exp = experiment::make_experiment(num_measurement_threads, *control_thread, *measurement_threads);
    LOG(INFO) << "===================== DONE ====";

    auto results = exp.run(
        metric,
        // get_connection
        [&](auto thread_idx) {
            auto conn = std::make_shared<thread_state>();

            conn->ch = pch->get_process()->make_channel(
                pch->get_config(), "clt-"+std::to_string(thread_idx)+"/"+std::to_string(num_measurement_threads))
                .get();

            for (size_t idx = 0; idx < max_in_flight; idx++) {
                CHECK(conn->available_idx.insert(idx).second);

                conn->resp_client[idx] = conn->ch->make_request_builder(
                    conn->ch->get_default_endpoint(),
                    [idx, conn=conn.get()](auto ch, auto args) {
                        conn->finish_experiment_cb(conn->start_time[idx]);
                        conn->available_idx.insert(idx);
                        conn->ch->break_run();
                    })
                    .on_channel()
                    .make_request()
                    .get();

                auto name = "resp_client_" + std::to_string(thread_idx) + "_" + std::to_string(idx);
                gns->publish_named(conn->ch, conn->resp_client[idx], name)
                    .then([&, idx](auto& fut) {
                        conn->gns_resp_client[idx] = fut.get();
                    })
                    .as_callback();
            }
            LOG(INFO) << "=== DONE ==================";

            for (size_t idx = 0; idx < max_in_flight; idx++) {
                auto name = "req_server_" + std::to_string(thread_idx) + "_" + std::to_string(idx);
                conn->req_server[idx] = gns->get_wait_for<cap::request>(conn->ch, name, 2min)
                    .get();
            }

            ready->run_until_ready();

            return conn;
        },
        // run_until
        [&](auto thread_idx, auto& conn, auto& finish_experiment, auto&& stop_cond) {
            conn->ch->run_until(stop_cond);
        },
        // stop_run_until
        [&](auto thread_idx, auto& conn) {
            conn->ch->break_run();
        },
        // start_experiment
        [&](auto thread_idx, auto& conn, auto start_time, auto& finish_experiment) {
            if (conn->available_idx.empty()) {
                conn->ch->run_until([&conn]() {
                    return not conn->available_idx.empty();
                });
            }

            size_t idx = conn->available_idx.extract(conn->available_idx.begin()).value();

            conn->start_time[idx] = start_time;
            conn->finish_experiment_cb = finish_experiment;

            conn->ch->invoke(conn->req_server[idx])
                .as_callback();
        });

    results.write_csv(output);
}
