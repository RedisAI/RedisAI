#include <string>
#include "torch_redis.h"
#include "../../redismodule.h"

torch::IValue IValueFromRedisReply(RedisModuleCallReply *reply){

    int reply_type = RedisModule_CallReplyType(reply);
    switch(reply_type) {
        case REDISMODULE_REPLY_NULL: {
            return torch::IValue();
        }
        case REDISMODULE_REPLY_STRING: {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            std::string str = replyStr;
            return torch::IValue(str.substr(0,len));
        }
        case REDISMODULE_REPLY_INTEGER: {
            int intValue = RedisModule_CallReplyInteger(reply);
            return torch::IValue(intValue);
        }
        case REDISMODULE_REPLY_ARRAY: {
            c10::impl::GenericList vec = c10::impl::GenericList(c10::AnyType::create());
            size_t len = RedisModule_CallReplyLength(reply);
            for (auto i = 0; i < len; ++i) {
                RedisModuleCallReply *subReply = RedisModule_CallReplyArrayElement(reply, i);
                torch::IValue value = IValueFromRedisReply(subReply);
                vec.push_back(value);
            }
            return torch::IValue(vec);
        }
        case REDISMODULE_REPLY_ERROR: {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            throw std::runtime_error(replyStr);
            break;
        }
        default:{
            throw(std::runtime_error("Unsupported redis type"));
        }
    }
}

torch::IValue redisExecute(std::string fn_name, std::vector<std::string> args ) {
  RedisModuleCtx* ctx = RedisModule_GetThreadSafeContext(NULL);
  size_t len = args.size();
  RedisModuleString* arguments[len];
  len = 0;
  for (std::vector<std::string>::iterator it = args.begin(); it != args.end(); it++) {
      const std::string arg = *it;
      const char* str = arg.c_str();
      arguments[len++] = RedisModule_CreateString(ctx, str, strlen(str));
  }

    RedisModuleCallReply *reply = RedisModule_Call(ctx, fn_name.c_str(), "v", arguments, len);
//   RedisValue value = RedisValue::fromRedisReply(RedisModule_Call(ctx, fn_name.c_str(), "v", arguments, len));
  torch::IValue value = IValueFromRedisReply(reply);
  RedisModule_FreeThreadSafeContext(ctx);
  RedisModule_FreeCallReply(reply);
  return value;
}
