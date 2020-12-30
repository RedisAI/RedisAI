#include "torch/jit.h"
#include "torch/script.h"
#include "torch/csrc/jit/frontend/resolver.h"

#include "torch_redis_value.h"

namespace torch {
    namespace jit {
        namespace script {
            struct RedisResolver : public Resolver {
                
                std::shared_ptr<SugaredValue> resolveValue(const std::string& name, Function& m, const SourceRange& loc) override {
                    if(strcasecmp(name.c_str(), "torch") == 0) {
                        return std::make_shared<BuiltinModule>("aten");
                    }
                    else if (strcasecmp(name.c_str(), "redis") == 0) {
                        return std::make_shared<BuiltinModule>("redis");
                    }
                    return nullptr;
                }

                TypePtr resolveType(const std::string& name, const SourceRange& loc) override {
                    return nullptr;
                }

            };
            inline std::shared_ptr<RedisResolver> redisResolver() {
                    return std::make_shared<RedisResolver>();
                }
        }
    }
}


// c10::intrusive_ptr<RedisValue> redisExecute(std::string fn_name, std::vector<std::string> args );

torch::IValue redisExecute(std::string fn_name, std::vector<std::string> args );



static auto registry = torch::RegisterOperators("redis::execute", &redisExecute);