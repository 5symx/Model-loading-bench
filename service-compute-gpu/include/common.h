#ifndef COMMON_H_
#define COMMON_H_
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib> 
#include <memory> //shared_ptr
#include <map>
#include <vector>
#include <iostream>
#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
#define PAGE_SIZE 2097152
#define RECV_BUF_SIZE (PAGE_SIZE/2)
#define SEND_BUF_SIZE (PAGE_SIZE/2)
#define REQS_COUNT 1
#define N_REGS 1
//lenet 28
#define IMG_DIMENSION 32
//lenet 320

//cliet-server params
#define N_REQUESTS 64 // 1000 is WAMUP
#define BURST N_REQUESTS  //N_REQUESTS
#define MAX_CLIENTS 4
#define WARMUP 0
//QP param
#define SLOTSNUM 10
#define CUDA_CHECK(f) do {                                                                  \
	cudaError_t e = f;                                                                      \
	if (e != cudaSuccess) {                                                                 \
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
		exit(1);                                                                            \
	}                                                                                       \
} while (0)
#define SQR(a) ((a) * (a))

//gpu parameters 
// we use BLOCKSNUM for gpu kernel parameter and for create queue pairs, each TB has its own queue pair.
#define THREADS_QUEUE 1024

enum device_opcode { VCA, GPU, SSD, DEVICES_NUM };
enum req_opcode { CREATE_SLICE,DESTROY_SLICE,SLICE_MEMORY_ALLOCATE,SLICE_REGISTER_FUNCTION,SLICE_CALL_FUNCTION, REQ_NUM};
enum func_opcode { NULL_FUNC, FACE_RECOGNITION, LENET, FACE_VERIFICATION, FUNC_NUM}; //FACE_RECOGNITION hardcoded with 1 , LENET hardcoded with 2

enum mode_opcode{
	NULL_MODE, PROGRAM_MODE_STREAMS , PROGRAM_MODE_QUEUE 
};

enum SERVICE_ERROR{SERVICE_SUCCESS,SERVICE_INVALID_SLICE_ID, SERVICE_INVALID_FUNC_ID,SERVICE_INVALID_SRV_PARAM,SERVICE_ERROR_NUM};

enum mem_alloc_type {ALLOC_HOST = 0, ALLOC_DEVICE = 1, ALLOC_MANAGED = 2};

typedef unsigned char uchar;
using namespace std;

double static inline get_time_usec(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6L + t.tv_nsec * 1e-3;
}

//request & response
///*****************************************************************
// ******************************************************************/
class Request{
	private:
		size_t sz;
	public:
		Request(size_t _sz){
			sz=_sz;
			func_args=(uchar**)malloc(sizeof(uchar*)*sz);
		}
		Request(){
			sz=0;
		}
		~Request(){
			//free(func_args);
		}
		uint64_t req_id; 	
		uint64_t slice_id;
		req_opcode opcode;
		//TODO call function
		func_opcode function_id; 
		//char* func_args[2];
		int num;
		uchar *images1;
		uchar *images2;
		uchar** func_args;
};

struct Response{
	uint64_t status_header;
	uint64_t req_id;
	uint64_t slice_id;
	//retured by register function
	uint64_t function_id;
	char* func_result;
	int num;
	double distance;
};
//memory
///*****************************************************************
// ******************************************************************/
struct memory{
	char* base_addr; //from shared memory
	size_t size;
};
//slice
///*****************************************************************
// ******************************************************************/
/*class slice{
	public:
		uint64_t slice_id;
		mode_opcode mode;
		cudaStream_t copyStream;
		queue_pairs* qp;

		std::vector<unsigned char*>* mems;
		std::map<uint64_t, uint64_t>* funcs;

		slice(uint64_t _slice_id,mode_opcode _mode){
			slice_id=_slice_id;
			mode=_mode;

			qp=new queue_pairs(NULL,
					SLOTSNUM,
					NULL,
					NULL,
					send_q_type,
					NULL, //TODO response
					SLOTSNUM,
					NULL,
					NULL,
					recv_q_type);

			mems=new std::vector<unsigned char*>();
			funcs=new std::map<uint64_t, uint64_t>();
		};
		~slice(){
			mems->clear();
			//delete qp;	
			delete mems;
			delete funcs;
		}
		int enqueue_resp(int _block_id,Request new_request,double* req_t_end,double* total_distance){


		}
		void push(Request req,int block_id){
			Queue<Request>* request_q_host=&(qp->send_q[block_id]);
			request_q_host->enqueue(req);
			qp->reqNum[block_id]++;

		}
		Response poll(int block_id,double* req_t_end,double* total_distance){
			Queue<Response>* response_q_host=&qp->recv_q[block_id];
			Response cur_response  =response_q_host->dequeue();
			req_t_end[cur_response.num] = get_time_usec();
			*total_distance += cur_response.distance;
			qp->respNum[block_id]++;
			return cur_response;

		}
		void service_end(){
			qp->end_connection();
		}

};*/
#endif /*COMMON_H_*/


