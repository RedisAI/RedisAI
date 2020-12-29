#pragma once
#include "torch/script.h"
#include "torch/custom_class.h"
#include "../../redismodule.h"

struct RedisValue : torch::CustomClassHolder {
  private:
    union {
        int intValue;
        std::string stringValue;
        std::vector<RedisValue *> arrayValue;
    };

  public:
    RedisValue(RedisModuleCallReply *reply) {
        // if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_ARRAY) {
        //     size_t len = RedisModule_CallReplyLength(reply);
        //     for (auto i = 0; i < len; ++i) {
        //         RedisModuleCallReply *subReply = RedisModule_CallReplyArrayElement(reply, i);
        //         RedisValue value(subReply);
        //         arrayValue.push_back(value);
        //     }
        //     return;
        // }

        if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING ||
            RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_ERROR) {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            stringValue= replyStr;
            return;

        }

        if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_INTEGER) {
            intValue = RedisModule_CallReplyInteger(reply);
            return;
        }

        throw(std::runtime_error("Unsupported redis type"));
    }

    virtual ~RedisValue() {

    }
};
