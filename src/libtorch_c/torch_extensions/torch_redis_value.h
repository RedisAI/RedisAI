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
        if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_ARRAY) {
            size_t len = RedisModule_CallReplyLength(reply);
            for (auto i = 0; i < len; ++i) {
                RedisModuleCallReply *subReply = RedisModule_CallReplyArrayElement(reply, i);
                RedisValue value(subReply);
                arrayValue.push_back(value);
            }
        }

        if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING ||
            RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_ERROR) {
            size_t len;
            const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
            PyObject *ret = PyUnicode_FromStringAndSize(replyStr, len);
            if (!ret) {
                PyErr_Clear();
                ret = PyByteArray_FromStringAndSize(replyStr, len);
            }
            return ret;
        }

        if (RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_INTEGER) {
            long long val = RedisModule_CallReplyInteger(reply);
            return PyLong_FromLongLong(val);
        }
    }
};
