#include "../../redismodule.h"
#include "../../util/arr.h"

#include "torch/csrc/jit/frontend/resolver.h"
#include "torch/script.h"
#include "torch/jit.h"

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

void redisExecute(std::string fn_name, std::vector<std::string> args ) {
  RedisModuleCtx* ctx = RedisModule_GetThreadSafeContext(NULL);
  size_t len = args.size();
  RedisModuleString* arguments[len];
  len = 0;
  for (std::vector<std::string>::iterator it = args.begin(); it != args.end(); it++) {
      const std::string arg = *it;
      const char* str = arg.c_str();
      arguments[len++] = RedisModule_CreateString(ctx, str, strlen(str));
  }
  RedisModule_Call(ctx, fn_name.c_str(), "v", arguments, len);
  RedisModule_FreeThreadSafeContext(ctx);
}

static auto registry = torch::RegisterOperators("redis::execute", &redisExecute);