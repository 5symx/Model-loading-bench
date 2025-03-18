#pragma once

#include <fractos/common/cmdline.hpp>
#include <fractos/common/cpu.hpp>
#include <fractos/common/experiment.hpp>

static inline
auto
options()
{
    using namespace fractos::common;

    auto odesc = cmdline::options();
    odesc.add_options()
        ("metric", cmdline::po::value<std::string>()->required(),
         "target metric (see fractos::common::experiment::parse_metric)")
        ("output", cmdline::po::value<std::string>(),
         "output file")
        ("control-thread", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+all+1"),
         "CPU set to pin the control thread to (see fractos::common::cpu::parse_set())")
        ("measurement-threads", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+all"),
         "CPU set to create and pin measurement threads (see fractos::common::cpu::parse_set())")
        ;
    return odesc;
}



static inline auto
parse(auto& odesc, auto argc, auto argv)
{
    using namespace fractos::common;

    auto [args, pch] = cmdline::parse(odesc, argc, argv);

    auto metric = experiment::parse_metric(args["metric"].template as<std::string>());
    // if (metric.get_name() == "latency" and GPROF || PERF) {
    //     LOG(INFO) << "setting fixed-length experiment";
    //     metric.params.latency.stddev_perc = 0;
    //     params.confidence_sigma = 0;
    //     params.batch_group_size = 1;
    //     params.batch_size_warmup = 0;
    //     params.batch_size = 20000000;
    // }

    std::string output = "";
    if (args.count("output")) {
        output = args["output"].template as<std::string>();
    }

    auto base = cpu::get_current_set();

    auto control_thread = cpu::parse_set(*base, args["control-thread"].template as<std::string>());
    if (control_thread->size() == 0) {
        std::cerr << "Error: empty cpuset for --control-thread" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Control thread: " << cpu::to_string(*control_thread);

    auto measurement_threads = cpu::parse_set(*base, args["measurement-threads"].template as<std::string>());
    if (measurement_threads->size() == 0) {
        std::cerr << "Error: empty cpuset for --measurement-threads" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Measurement threads: " << cpu::to_string(*measurement_threads);

    return std::make_tuple(args, pch, output, metric, control_thread, measurement_threads);
}



struct ready_info {
    ready_info(std::shared_ptr<fractos::core::gns::service> gns, std::shared_ptr<fractos::core::channel> ch);

    static void send_ready(std::shared_ptr<fractos::core::gns::service> gns,
                           std::shared_ptr<fractos::core::channel> ch,
                           size_t max_threads);
    void run_until_ready();

    std::atomic<size_t> cur;
    std::atomic<size_t> max;

    std::shared_ptr<fractos::core::gns::service> gns;
    std::shared_ptr<fractos::core::channel> ch;
    fractos::core::cap::request req_ready;
    fractos::core::gns::guard guard_ready;
};

ready_info::ready_info(std::shared_ptr<fractos::core::gns::service> gns,
                       std::shared_ptr<fractos::core::channel> ch)
    :cur(0)
    ,max(0)
    ,gns(gns)
    ,ch(ch)
{
    req_ready = ch->make_request_builder(
        ch->get_default_endpoint(),
        [this](auto ch, auto args) {
            CHECK(args->imms_size() == sizeof(size_t));
            CHECK(args->caps_count() == 0);

            auto args_max = *(size_t*)&args->imms_raw[0];
            CHECK((max == 0) or (max = args_max));
            max = args_max;
            cur++;
            ch->break_run();
        })
        .on_channel()
        .make_request()
        .get();
    guard_ready = gns->publish_named(ch, req_ready, "server_ready")
        .get();
}

void
ready_info::send_ready(std::shared_ptr<fractos::core::gns::service> gns,
                       std::shared_ptr<fractos::core::channel> ch,
                       size_t max_threads)
{
    using namespace std::chrono_literals;

    auto req = gns->get_wait_for<fractos::core::cap::request>(ch, "server_ready", 0min)
        .get();
    ch->make_request_builder(req)
        .set_imm(0, &max_threads, sizeof(max_threads))
        .on_channel()
        .invoke()
        .get();
}

void
ready_info::run_until_ready()
{
    ch->run_until([this]() {
        return max and max == cur;
    });
}
