/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ragent.ai.rag.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.ragent.ai.framework.context.UserContext;
import com.ragent.ai.framework.convention.ChatMessage;
import com.ragent.ai.framework.convention.ChatRequest;
import com.ragent.ai.framework.trace.RagTraceContext;
import com.ragent.ai.infra.chat.LLMService;
import com.ragent.ai.infra.chat.StreamCallback;
import com.ragent.ai.infra.chat.StreamCancellationHandle;
import com.ragent.ai.rag.aop.ChatRateLimit;
import com.ragent.ai.rag.core.guidance.GuidanceDecision;
import com.ragent.ai.rag.core.guidance.IntentGuidanceService;
import com.ragent.ai.rag.core.intent.IntentResolver;
import com.ragent.ai.rag.core.memory.ConversationMemoryService;
import com.ragent.ai.rag.core.prompt.PromptContext;
import com.ragent.ai.rag.core.prompt.PromptTemplateLoader;
import com.ragent.ai.rag.core.prompt.RAGPromptService;
import com.ragent.ai.rag.core.retrieve.RetrievalEngine;
import com.ragent.ai.rag.core.rewrite.QueryRewriteService;
import com.ragent.ai.rag.core.rewrite.RewriteResult;
import com.ragent.ai.rag.dto.IntentGroup;
import com.ragent.ai.rag.dto.RetrievalContext;
import com.ragent.ai.rag.dto.SubQuestionIntent;
import com.ragent.ai.rag.service.RAGChatService;
import com.ragent.ai.rag.service.handler.StreamCallbackFactory;
import com.ragent.ai.rag.service.handler.StreamTaskManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;

import static com.ragent.ai.rag.constant.RAGConstant.CHAT_SYSTEM_PROMPT_PATH;
import static com.ragent.ai.rag.constant.RAGConstant.DEFAULT_TOP_K;

/**
 * RAG 对话服务默认实现
 * <p>
 * 核心流程：
 * 记忆加载 -> 改写拆分 -> 意图解析 -> 歧义引导 -> 检索(MCP + KB) -> Prompt 组装 -> 流式输出
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RAGChatServiceImpl implements RAGChatService {

    private final LLMService llmService;
    private final RAGPromptService promptBuilder;
    private final PromptTemplateLoader promptTemplateLoader;
    private final ConversationMemoryService memoryService;
    private final StreamTaskManager taskManager;
    private final IntentGuidanceService guidanceService;
    private final StreamCallbackFactory callbackFactory;
    private final QueryRewriteService queryRewriteService;
    private final IntentResolver intentResolver;
    private final RetrievalEngine retrievalEngine;

    @Override
    @ChatRateLimit
    public void streamChat(String question, String conversationId, Boolean deepThinking, SseEmitter emitter) {
        String actualConversationId = StrUtil.isBlank(conversationId) ? IdUtil.getSnowflakeNextIdStr() : conversationId;
        String taskId = StrUtil.isBlank(RagTraceContext.getTaskId())
                ? IdUtil.getSnowflakeNextIdStr()
                : RagTraceContext.getTaskId();
        log.info("开始流式对话，会话 ID：{}，任务 ID：{}", actualConversationId, taskId);
        boolean thinkingEnabled = Boolean.TRUE.equals(deepThinking);

        StreamCallback callback = callbackFactory.createChatEventHandler(emitter, actualConversationId, taskId);
        try {
            String userId = UserContext.getUserId();
            List<ChatMessage> history = memoryService.loadAndAppend(actualConversationId, userId, ChatMessage.user(question));

            RewriteResult rewriteResult = queryRewriteService.rewriteWithSplit(question, history);
            List<SubQuestionIntent> subIntents = intentResolver.resolve(rewriteResult);

            GuidanceDecision guidanceDecision = guidanceService.detectAmbiguity(rewriteResult.rewrittenQuestion(), subIntents);
            if (guidanceDecision.isPrompt()) {
                callback.onContent(guidanceDecision.getPrompt());
                callback.onComplete();
                return;
            }

            boolean allSystemOnly = subIntents.stream()
                    .allMatch(si -> intentResolver.isSystemOnly(si.nodeScores()));
            if (allSystemOnly) {
                String customPrompt = subIntents.stream()
                        .flatMap(si -> si.nodeScores().stream())
                        .map(ns -> ns.getNode().getPromptTemplate())
                        .filter(StrUtil::isNotBlank)
                        .findFirst()
                        .orElse(null);
                StreamCancellationHandle handle = streamSystemResponse(rewriteResult.rewrittenQuestion(), history, customPrompt, callback);
                taskManager.bindHandle(taskId, handle);
                return;
            }

            RetrievalContext ctx = retrievalEngine.retrieve(subIntents, DEFAULT_TOP_K);
            if (ctx.isEmpty()) {
                String emptyReply = "未检索到与问题相关的文档内容。";
                callback.onContent(emptyReply);
                callback.onComplete();
                return;
            }

            // 聚合所有意图用于 prompt 规划
            IntentGroup mergedGroup = intentResolver.mergeIntentGroup(subIntents);

            StreamCancellationHandle handle = streamLLMResponse(
                    rewriteResult,
                    ctx,
                    mergedGroup,
                    history,
                    thinkingEnabled,
                    callback
            );
            taskManager.bindHandle(taskId, handle);
        } catch (Exception e) {
            log.error("流式对话处理异常，会话 ID：{}，任务 ID：{}", actualConversationId, taskId, e);
            callback.onError(e);
        }
    }

    @Override
    public void stopTask(String taskId) {
        taskManager.cancel(taskId);
    }

    // ==================== LLM 响应 ====================

    private StreamCancellationHandle streamSystemResponse(String question, List<ChatMessage> history,
                                                          String customPrompt, StreamCallback callback) {
        String systemPrompt = StrUtil.isNotBlank(customPrompt)
                ? customPrompt
                : promptTemplateLoader.load(CHAT_SYSTEM_PROMPT_PATH);

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(systemPrompt));
        if (CollUtil.isNotEmpty(history)) {
            messages.addAll(history);
        }
        messages.add(ChatMessage.user(question));

        ChatRequest req = ChatRequest.builder()
                .messages(messages)
                .temperature(0.7D)
                .thinking(false)
                .build();
        return llmService.streamChat(req, callback);
    }

    private StreamCancellationHandle streamLLMResponse(RewriteResult rewriteResult, RetrievalContext ctx,
                                                       IntentGroup intentGroup, List<ChatMessage> history,
                                                       boolean deepThinking, StreamCallback callback) {
        PromptContext promptContext = PromptContext.builder()
                .question(rewriteResult.rewrittenQuestion())
                .mcpContext(ctx.getMcpContext())
                .kbContext(ctx.getKbContext())
                .mcpIntents(intentGroup.mcpIntents())
                .kbIntents(intentGroup.kbIntents())
                .intentChunks(ctx.getIntentChunks())
                .build();

        List<ChatMessage> messages = promptBuilder.buildStructuredMessages(
                promptContext,
                history,
                rewriteResult.rewrittenQuestion(),
                rewriteResult.subQuestions()  // 传入子问题列表
        );
        ChatRequest chatRequest = ChatRequest.builder()
                .messages(messages)
                .thinking(deepThinking)
                .temperature(ctx.hasMcp() ? 0.3D : 0D)  // MCP 场景稍微放宽温度
                .topP(ctx.hasMcp() ? 0.8D : 1D)
                .build();

        return llmService.streamChat(chatRequest, callback);
    }
}
