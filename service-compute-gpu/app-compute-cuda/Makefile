CPPFLAGS =
CPPFLAGS += -I$(VPATH)
CPPFLAGS += -I/usr/local/cuda/include/

CXXFLAGS = -g -std=c++17 -Wall -fconcepts -Werror
CXXFLAGS += -MT $@ -MMD -MP -MF $(CURDIR)/.$*.d
CXXFLAGS += $(shell pkg-config fractos-service-compute-cuda --cflags)
CXXFLAGS += $(shell pkg-config fractos-common --cflags)
CXXFLAGS += $(shell pkg-config fractos-core --cflags)
ifeq ($(BUILD_FLAG_production),1)
CXXFLAGS += -Ofast -DNDEBUG
endif
ifeq ($(BUILD_FLAG_debug),1)
CXXFLAGS += -O0
endif
ifeq ($(BUILD_FLAG_gprof),1)
CXXFLAGS += -pg
CXXFLAGS += -DGPROF=1
endif

LDFLAGS =
LDFLAGS += $(shell pkg-config fractos-service-compute-cuda --libs)
LDFLAGS += $(shell pkg-config fractos-common --libs)
LDFLAGS += $(shell pkg-config fractos-core --libs)
LDFLAGS += -lglog -lpthread -libverbs
LDFLAGS += -lboost_program_options -lboost_system -lboost_thread
ifeq ($(BUILD_FLAG_gprof),1)
LDFLAGS += -pg
endif


LDFLAGS += -L/usr/local/cuda/lib64 -lcudart  -lcuda -L"/usr/local/cuda/lib64/stubs/" -lnvToolsExt



all: simple rpc_manual
# all: pipeline

#simple: simple.cpp
%: %.cpp
ifeq ($(V),1)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) 
else
	@echo "  CXXLD    $@"
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
endif





-include $(wildcard $(CURDIR)/.*.d)
