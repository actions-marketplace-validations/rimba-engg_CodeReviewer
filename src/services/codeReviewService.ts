import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from 'langchain/prompts'
import { LLMChain } from 'langchain/chains'
import { BaseChatModel } from 'langchain/dist/chat_models/base'
import type { ChainValues } from 'langchain/dist/schema'
import { PullRequestFile } from './pullRequestService'
import parseDiff from 'parse-diff'
import { LanguageDetectionService } from './languageDetectionService'
import { exponentialBackoffWithJitter } from '../httpUtils'
import { Effect, Context } from 'effect'
import { NoSuchElementException, UnknownException } from 'effect/Cause'

export interface CodeReviewService {
  codeReviewFor(
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService>
  codeReviewForChunks(
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService>
}

export const CodeReviewService = Context.GenericTag<CodeReviewService>('CodeReviewService')

export class CodeReviewServiceImpl {
  private llm: BaseChatModel
  private chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(`
<CONTEXT>
  You are an experienced software engineer tasked with reviewing a Pull Request. 
  Your goal is to provide concise, actionable feedback that helps improve code quality while highlighting any critical issues.
</CONTEXT>
    
<EXAMPLE_OUTPUT>
  <THINKING_OUT_LOUD>
  [Your detailed analysis and observations]
  </THINKING_OUT_LOUD>
  
  <KEY_ISSUES>
  1. [Issue description]
     ```[language]
     [Code suggestion]
     ```
  
  2. [Issue description]
     ```[language]
     [Code suggestion]
     ```
  </KEY_ISSUES>
  
  <RATINGS>
  - Code Quality: ⭐⭐☆
  - Maintainability: ⭐⭐⭐
  - Readability: ⭐⭐☆
  - Performance: ⭐⭐⭐
  - Security: ⭐☆☆
  </RATINGS>
  
  <MAJOR_FLAGS>
  **⚠️ [Description of critical issue, if any]**
  </MAJOR_FLAGS>
</EXAMPLE_OUTPUT>

<NOTES>
- PLS MAKE SURE TO INCLUDE <RATINGS> SECTION WITH STARS ⭐
- PLS MAKE SURE TO INCLUDE <MAJOR_FLAGS> SECTION
- KEEP FEEDBACK CONCISE AND ACTIONABLE
</NOTES>`)
    HumanMessagePromptTemplate.fromTemplate(`Here's the git diff you need to review:
<GIT_DIFF>
{diff}
</GIT_DIFF>

The programming language used in this diff is:
<LANGUAGE>
{lang}
</LANGUAGE>

<INSTRUCTIONS>
1. Analyze the git diff carefully.
2. In your analysis, consider the following aspects:
   - Code quality
   - Maintainability
   - Readability
   - Performance
   - Security
   - Potential bugs or vulnerabilities

3. Prioritize your findings, focusing on the most important issues first.
4. Provide concise feedback with specific code suggestions where applicable.
5. Rate each aspect (code quality, maintainability, readability, performance, security) on a scale of 1 to 3 stars.
6. Flag any major bugs or blatant failures prominently.
</INSTRUCTIONS>

<OUTPUT_FORMAT>
Use GitHub Markdown format for your response. Structure your review as follows:
1. **Analysis**: Show your thought process and observations inside <code_review_process> tags.
2. **Key Issues**: List the most important issues you've identified, with code suggestions where applicable.
3. **Minor Improvements**: Briefly mention any minor issues or suggestions.
4. **Ratings**: Provide star ratings for each aspect.
5. **Major Flags**: If applicable, prominently flag any critical issues.

Before providing your final output, break down your review process inside <code_review_process> tags:
- List the files changed in the diff
- For each file, summarize the changes made
- Identify potential issues in each aspect (code quality, maintainability, readability, performance, security)
- Categorize issues as major or minor
- Suggest improvements for each issue
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
  <THINKING_OUT_LOUD>
  [Your detailed analysis and observations]
  </THINKING_OUT_LOUD>
  
  <KEY_ISSUES>
  1. [Issue description]
     ```[language]
     [Code suggestion]
     ```
  
  2. [Issue description]
     ```[language]
     [Code suggestion]
     ```
  </KEY_ISSUES>
  
  <RATINGS>
  - Code Quality: ⭐⭐☆
  - Maintainability: ⭐⭐⭐
  - Readability: ⭐⭐☆
  - Performance: ⭐⭐⭐
  - Security: ⭐☆☆
  </RATINGS>
  
  <MAJOR_FLAGS>
  **⚠️ [Description of critical issue, if any]**
  </MAJOR_FLAGS>
</EXAMPLE_OUTPUT>

<NOTES>
- PLS MAKE SURE TO INCLUDE <RATINGS> SECTION WITH STARS ⭐
- PLS MAKE SURE TO INCLUDE <MAJOR_FLAGS> SECTION
- KEEP FEEDBACK CONCISE AND ACTIONABLE
</NOTES>
`)
  ])
  private chain: LLMChain<string>

  constructor(llm: BaseChatModel) {
    this.llm = llm
    this.chain = new LLMChain({
      prompt: this.chatPrompt,
      llm: this.llm
    })
  }

  codeReviewFor = (
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService> =>
    LanguageDetectionService.pipe(
      Effect.flatMap(languageDetectionService => languageDetectionService.detectLanguage(file.filename)),
      Effect.flatMap(lang =>
        Effect.retry(
          Effect.tryPromise(() => this.chain.call({ lang, diff: file.patch })),
          exponentialBackoffWithJitter(3)
        )
      )
    )

  codeReviewForChunks(
    file: PullRequestFile
  ): Effect.Effect<ChainValues[], NoSuchElementException | UnknownException, LanguageDetectionService> {
    const programmingLanguage = LanguageDetectionService.pipe(
      Effect.flatMap(languageDetectionService => languageDetectionService.detectLanguage(file.filename))
    )
    const fileDiff = Effect.sync(() => parseDiff(file.patch)[0])

    return Effect.all([programmingLanguage, fileDiff]).pipe(
      Effect.flatMap(([lang, fd]) =>
        Effect.all(fd.chunks.map(chunk => Effect.tryPromise(() => this.chain.call({ lang, diff: chunk.content }))))
      )
    )
  }
}
