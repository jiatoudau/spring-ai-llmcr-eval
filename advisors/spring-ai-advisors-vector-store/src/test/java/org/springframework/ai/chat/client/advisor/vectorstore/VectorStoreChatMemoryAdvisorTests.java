/*
 * Copyright 2023-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.chat.client.advisor.vectorstore;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.scheduler.Scheduler;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link VectorStoreChatMemoryAdvisor}.
 *
 * @author Thomas Vitale
 */
@ExtendWith(MockitoExtension.class)
class VectorStoreChatMemoryAdvisorTests {

	@Mock
	ChatModel chatModel;

	@Mock
	VectorStore vectorStore;

	@Captor
	ArgumentCaptor<SearchRequest> searchRequestCaptor;

	@Captor
	ArgumentCaptor<List<Document>> documentsCaptor;

	@Test
	void whenVectorStoreIsNullThenThrow() {
		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("vectorStore cannot be null");
	}

	@Test
	void whenDefaultConversationIdIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenDefaultConversationIdIsEmptyThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenSchedulerIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).scheduler(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("scheduler cannot be null");
	}

	@Test
	void whenSystemPromptTemplateIsNullThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).systemPromptTemplate(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("systemPromptTemplate cannot be null");
	}

	@Test
	void whenDefaultTopKIsZeroThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(0).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("topK must be greater than 0");
	}

	@Test
	void whenDefaultTopKIsNegativeThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(-1).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("topK must be greater than 0");
	}

	@Test
	void whenBuilderWithValidVectorStoreThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore).build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithAllValidParametersThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		Scheduler scheduler = Mockito.mock(Scheduler.class);
		PromptTemplate systemPromptTemplate = Mockito.mock(PromptTemplate.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("test-conversation")
			.scheduler(scheduler)
			.systemPromptTemplate(systemPromptTemplate)
			.defaultTopK(5)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenDefaultConversationIdIsBlankThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId("   ").build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenBuilderWithValidConversationIdThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("valid-id")
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithValidTopKThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.defaultTopK(10)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithMinimumTopKThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(1).build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithLargeTopKThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.defaultTopK(1000)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderCalledMultipleTimesWithSameVectorStoreThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor1 = VectorStoreChatMemoryAdvisor.builder(vectorStore).build();
		VectorStoreChatMemoryAdvisor advisor2 = VectorStoreChatMemoryAdvisor.builder(vectorStore).build();

		assertThat(advisor1).isNotNull();
		assertThat(advisor2).isNotNull();
		assertThat(advisor1).isNotSameAs(advisor2);
	}

	@Test
	void whenBuilderWithCustomSchedulerThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		Scheduler customScheduler = Mockito.mock(Scheduler.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.scheduler(customScheduler)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithCustomSystemPromptTemplateThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		PromptTemplate customTemplate = Mockito.mock(PromptTemplate.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.systemPromptTemplate(customTemplate)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithEmptyStringConversationIdThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId("").build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenBuilderWithWhitespaceOnlyConversationIdThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId("\t\n\r ").build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenBuilderWithSpecialCharactersInConversationIdThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("conversation-id_123@domain.com")
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithMaxIntegerTopKThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.defaultTopK(Integer.MAX_VALUE)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithNegativeTopKThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore).defaultTopK(-100).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("topK must be greater than 0");
	}

	@Test
	void whenBuilderChainedWithAllParametersThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		Scheduler scheduler = Mockito.mock(Scheduler.class);
		PromptTemplate systemPromptTemplate = Mockito.mock(PromptTemplate.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("chained-test")
			.defaultTopK(42)
			.scheduler(scheduler)
			.systemPromptTemplate(systemPromptTemplate)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderParametersSetInDifferentOrderThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		Scheduler scheduler = Mockito.mock(Scheduler.class);
		PromptTemplate systemPromptTemplate = Mockito.mock(PromptTemplate.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.systemPromptTemplate(systemPromptTemplate)
			.defaultTopK(7)
			.scheduler(scheduler)
			.conversationId("order-test")
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderWithOverriddenParametersThenUseLastValue() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("first-id")
			.conversationId("second-id") // This should override the first
			.defaultTopK(5)
			.defaultTopK(10) // This should override the first
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderReusedThenCreatesSeparateInstances() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		// Simulate builder reuse (if the builder itself is stateful)
		var builder = VectorStoreChatMemoryAdvisor.builder(vectorStore).conversationId("shared-config");

		VectorStoreChatMemoryAdvisor advisor1 = builder.build();
		VectorStoreChatMemoryAdvisor advisor2 = builder.build();

		assertThat(advisor1).isNotNull();
		assertThat(advisor2).isNotNull();
		assertThat(advisor1).isNotSameAs(advisor2);
	}

	@Test
	void whenBuilderWithLongConversationIdThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);
		String longId = "a".repeat(1000); // 1000 character conversation ID

		VectorStoreChatMemoryAdvisor advisor = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId(longId)
			.build();

		assertThat(advisor).isNotNull();
	}

	@Test
	void whenBuilderCalledWithNullAfterValidValueThenThrow() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		assertThatThrownBy(() -> VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.conversationId("valid-id")
			.conversationId(null) // Set to null after valid value
			.build()).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("defaultConversationId cannot be null or empty");
	}

	@Test
	void whenBuilderWithTopKBoundaryValuesThenSuccess() {
		VectorStore vectorStore = Mockito.mock(VectorStore.class);

		// Test with value 1 (minimum valid)
		VectorStoreChatMemoryAdvisor advisor1 = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.defaultTopK(1)
			.build();

		// Test with a reasonable upper bound
		VectorStoreChatMemoryAdvisor advisor2 = VectorStoreChatMemoryAdvisor.builder(vectorStore)
			.defaultTopK(10000)
			.build();

		assertThat(advisor1).isNotNull();
		assertThat(advisor2).isNotNull();
	}

	// -------------------------------------------------------------------------
	// Behavior: filter expression format (new string-based filter)
	// -------------------------------------------------------------------------

	@Test
	void whenBeforeCalledThenFilterExpressionUsesConversationIdFromContext() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("default-conv")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "context-conv-id"))
			.call()
			.content();

		SearchRequest captured = this.searchRequestCaptor.getValue();
		assertThat(captured.getFilterExpression()).isNotNull();
		assertThat(captured.getFilterExpression().toString()).contains("context-conv-id");
	}

	@Test
	void whenBeforeCalledWithNoContextConversationIdThenFilterUsesDefaultConversationId() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("my-default-conversation")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.call()
			.content();

		SearchRequest captured = this.searchRequestCaptor.getValue();
		assertThat(captured.getFilterExpression()).isNotNull();
		assertThat(captured.getFilterExpression().toString()).contains("my-default-conversation");
	}

	@Test
	void whenBeforeCalledThenContextConversationIdOverridesDefault() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("builder-default")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "runtime-override"))
			.call()
			.content();

		SearchRequest captured = this.searchRequestCaptor.getValue();
		assertThat(captured.getFilterExpression().toString()).contains("runtime-override");
		assertThat(captured.getFilterExpression().toString()).doesNotContain("builder-default");
	}

	// -------------------------------------------------------------------------
	// Behavior: topK in SearchRequest
	// -------------------------------------------------------------------------

	@Test
	void whenBeforeCalledThenSearchRequestUsesDefaultTopK() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).defaultTopK(7).build();

		ChatClient.builder(this.chatModel).defaultAdvisors(advisor).build().prompt().user("hello").call().content();

		assertThat(this.searchRequestCaptor.getValue().getTopK()).isEqualTo(7);
	}

	@Test
	void whenBeforeCalledWithTopKInContextThenSearchRequestUsesContextTopK() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).defaultTopK(5).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.advisors(a -> a.param(VectorStoreChatMemoryAdvisor.TOP_K, "12"))
			.call()
			.content();

		assertThat(this.searchRequestCaptor.getValue().getTopK()).isEqualTo(12);
	}

	// -------------------------------------------------------------------------
	// Behavior: user message written to vector store
	// -------------------------------------------------------------------------

	@Test
	void whenBeforeCalledThenUserMessageWrittenToVectorStore() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("test-conv")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("my user query")
			.call()
			.content();

		// verify vectorStore.write() is called (once for user message, once for
		// assistant)
		verify(this.vectorStore, org.mockito.Mockito.atLeastOnce()).write(anyList());
	}

	@Test
	void whenBeforeCalledThenUserDocumentHasCorrectConversationIdMetadata() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("conv-meta-test")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("test message")
			.call()
			.content();

		verify(this.vectorStore, org.mockito.Mockito.atLeastOnce()).write(this.documentsCaptor.capture());
		List<Document> writtenDocs = this.documentsCaptor.getAllValues()
			.stream()
			.flatMap(List::stream)
			.filter(d -> "test message".equals(d.getText()))
			.toList();
		assertThat(writtenDocs).isNotEmpty();
		assertThat(writtenDocs.get(0).getMetadata()).containsEntry("conversationId", "conv-meta-test");
	}

	// -------------------------------------------------------------------------
	// Behavior: plain text memory (no XML escaping)
	// -------------------------------------------------------------------------

	@Test
	void whenDocumentsReturnedThenSystemPromptContainsPlainDocumentText() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		ArgumentCaptor<org.springframework.ai.chat.prompt.Prompt> promptCaptor = ArgumentCaptor
			.forClass(org.springframework.ai.chat.prompt.Prompt.class);
		given(this.chatModel.call(promptCaptor.capture())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));

		String docText = "Plain text memory content";
		given(this.vectorStore.similaritySearch(any(SearchRequest.class)))
			.willReturn(List.of(Document.builder().text(docText).build()));

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.call()
			.content();

		String systemText = promptCaptor.getValue().getInstructions().get(0).getText();
		assertThat(systemText).contains(docText);
	}

	@Test
	void whenDocumentTextContainsXmlCharactersThenSystemPromptContainsThemUnescaped() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		ArgumentCaptor<org.springframework.ai.chat.prompt.Prompt> promptCaptor = ArgumentCaptor
			.forClass(org.springframework.ai.chat.prompt.Prompt.class);
		given(this.chatModel.call(promptCaptor.capture())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("ok"))),
						ChatResponseMetadata.builder().build()));

		// XML characters should NOT be escaped since escaping was removed in this PR
		String textWithXml = "AT&T said <hello>";
		given(this.vectorStore.similaritySearch(any(SearchRequest.class)))
			.willReturn(List.of(Document.builder().text(textWithXml).build()));

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("query")
			.call()
			.content();

		String systemText = promptCaptor.getValue().getInstructions().get(0).getText();
		// The text should appear as-is (unescaped) since XML escaping was removed
		assertThat(systemText).contains("AT&T said <hello>");
		assertThat(systemText).doesNotContain("&amp;");
		assertThat(systemText).doesNotContain("&lt;");
	}

	@Test
	void whenMultipleDocumentsReturnedThenTheyAreJoinedInSystemPrompt() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		ArgumentCaptor<org.springframework.ai.chat.prompt.Prompt> promptCaptor = ArgumentCaptor
			.forClass(org.springframework.ai.chat.prompt.Prompt.class);
		given(this.chatModel.call(promptCaptor.capture())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("ok"))),
						ChatResponseMetadata.builder().build()));

		given(this.vectorStore.similaritySearch(any(SearchRequest.class)))
			.willReturn(List.of(Document.builder().text("first memory").build(),
					Document.builder().text("second memory").build()));

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("query")
			.call()
			.content();

		String systemText = promptCaptor.getValue().getInstructions().get(0).getText();
		assertThat(systemText).contains("first memory");
		assertThat(systemText).contains("second memory");
	}

	@Test
	void whenNullDocumentsReturnedThenSystemPromptHasEmptyMemory() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		ArgumentCaptor<org.springframework.ai.chat.prompt.Prompt> promptCaptor = ArgumentCaptor
			.forClass(org.springframework.ai.chat.prompt.Prompt.class);
		given(this.chatModel.call(promptCaptor.capture())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("ok"))),
						ChatResponseMetadata.builder().build()));

		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(null);

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("query")
			.call()
			.content();

		String systemText = promptCaptor.getValue().getInstructions().get(0).getText();
		// LONG_TERM_MEMORY section is present but has no documents listed
		assertThat(systemText).contains("LONG_TERM_MEMORY");
	}

	@Test
	void whenEmptyDocumentsReturnedThenSystemPromptHasEmptyMemory() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		ArgumentCaptor<org.springframework.ai.chat.prompt.Prompt> promptCaptor = ArgumentCaptor
			.forClass(org.springframework.ai.chat.prompt.Prompt.class);
		given(this.chatModel.call(promptCaptor.capture())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("ok"))),
						ChatResponseMetadata.builder().build()));

		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("query")
			.call()
			.content();

		String systemText = promptCaptor.getValue().getInstructions().get(0).getText();
		assertThat(systemText).contains("LONG_TERM_MEMORY");
	}

	// -------------------------------------------------------------------------
	// Behavior: getOrder and getScheduler
	// -------------------------------------------------------------------------

	@Test
	void whenBuilderWithCustomOrderThenGetOrderReturnsIt() {
		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).order(42).build();

		assertThat(advisor.getOrder()).isEqualTo(42);
	}

	@Test
	void whenBuilderWithDefaultOrderThenGetOrderReturnsAdvisorDefault() {
		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		// Default order is Advisor.DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER
		assertThat(advisor.getOrder()).isEqualTo(org.springframework.ai.chat.client.advisor.api.Advisor.DEFAULT_CHAT_MEMORY_PRECEDENCE_ORDER);
	}

	@Test
	void whenBuilderWithCustomSchedulerThenGetSchedulerReturnsIt() {
		Scheduler customScheduler = Mockito.mock(Scheduler.class);
		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).scheduler(customScheduler).build();

		assertThat(advisor.getScheduler()).isSameAs(customScheduler);
	}

	// -------------------------------------------------------------------------
	// Behavior: after() method writes assistant messages
	// -------------------------------------------------------------------------

	@Test
	void whenAfterCalledThenAssistantMessageWrittenToVectorStore() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("assistant response text"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("after-test-conv")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.call()
			.content();

		verify(this.vectorStore, org.mockito.Mockito.atLeastOnce()).write(this.documentsCaptor.capture());
		List<Document> assistantDocs = this.documentsCaptor.getAllValues()
			.stream()
			.flatMap(List::stream)
			.filter(d -> "assistant response text".equals(d.getText()))
			.toList();
		assertThat(assistantDocs).isNotEmpty();
		assertThat(assistantDocs.get(0).getMetadata()).containsEntry("conversationId", "after-test-conv");
	}

	@Test
	void whenAfterCalledWithContextConversationIdThenAssistantDocHasContextId() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("assistant reply"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("default-id")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "context-provided-id"))
			.call()
			.content();

		verify(this.vectorStore, org.mockito.Mockito.atLeastOnce()).write(this.documentsCaptor.capture());
		List<Document> assistantDocs = this.documentsCaptor.getAllValues()
			.stream()
			.flatMap(List::stream)
			.filter(d -> "assistant reply".equals(d.getText()))
			.toList();
		assertThat(assistantDocs).isNotEmpty();
		assertThat(assistantDocs.get(0).getMetadata()).containsEntry("conversationId", "context-provided-id");
	}

	// -------------------------------------------------------------------------
	// Behavior: search query uses user message text
	// -------------------------------------------------------------------------

	@Test
	void whenBeforeCalledThenSearchRequestQueryIsUserMessageText() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(new Generation(new AssistantMessage("answer"))),
						ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(this.searchRequestCaptor.capture())).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore).build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("what is the capital of France?")
			.call()
			.content();

		assertThat(this.searchRequestCaptor.getValue().getQuery()).isEqualTo("what is the capital of France?");
	}

	// -------------------------------------------------------------------------
	// Behavior: no writes for empty response
	// -------------------------------------------------------------------------

	@Test
	void whenAfterCalledWithNullChatResponseThenNoDocumentsWritten() {
		when(this.chatModel.getDefaultOptions()).thenReturn(ChatOptions.builder().build());
		// ChatResponse with empty results list
		given(this.chatModel.call(any())).willReturn(
				new ChatResponse(List.of(), ChatResponseMetadata.builder().build()));
		given(this.vectorStore.similaritySearch(any(SearchRequest.class))).willReturn(List.of());

		var advisor = VectorStoreChatMemoryAdvisor.builder(this.vectorStore)
			.conversationId("empty-response-test")
			.build();

		ChatClient.builder(this.chatModel)
			.defaultAdvisors(advisor)
			.build()
			.prompt()
			.user("hello")
			.call()
			.content();

		// Only the user message write should happen (from before()), not assistant
		verify(this.vectorStore, org.mockito.Mockito.atLeastOnce()).write(this.documentsCaptor.capture());
		List<Document> assistantDocs = this.documentsCaptor.getAllValues()
			.stream()
			.flatMap(List::stream)
			.filter(d -> d.getMetadata() != null && "ASSISTANT".equals(d.getMetadata().get("messageType")))
			.toList();
		assertThat(assistantDocs).isEmpty();
	}

}
