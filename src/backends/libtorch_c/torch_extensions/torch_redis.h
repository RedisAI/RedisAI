#include "torch/jit.h"
#include "torch/script.h"
#include "torch/csrc/jit/frontend/resolver.h"

namespace torch {
namespace jit {
namespace script {
struct RedisResolver : public Resolver {

    std::shared_ptr<SugaredValue> resolveValue(const std::string &name, GraphFunction &m,
                                               const SourceRange &loc) override {
        if (strcasecmp(name.c_str(), "torch") == 0) {
            return std::make_shared<BuiltinModule>("aten");
        } else if (strcasecmp(name.c_str(), "redis") == 0) {
            return std::make_shared<BuiltinModule>("redis");
        } else if (strcasecmp(name.c_str(), "redisAI") == 0) {
            return std::make_shared<BuiltinModule>("redisAI");
        }
        return nullptr;
    }

    TypePtr resolveType(const std::string &name, const SourceRange &loc) override {
        return nullptr;
    }
};
inline std::shared_ptr<RedisResolver> redisResolver() { return std::make_shared<RedisResolver>(); }
} // namespace script
} // namespace jit
} // namespace torch

torch::IValue redisExecute(const std::string &fn_name, const std::vector<std::string> &args);
torch::List<torch::IValue> asList(const torch::IValue &v);
std::vector<torch::Tensor> modelExecute(const std::string &model_key,
                                        const std::vector<torch::Tensor> &inputs,
                                        int64_t num_outputs);

// Register Redis and RedisAI costume ops in torch
void registerRedisOps(void);
