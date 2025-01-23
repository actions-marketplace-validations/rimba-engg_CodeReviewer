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
    HumanMessagePromptTemplate.fromTemplate(`You are an experienced software engineer tasked with reviewing a Pull Request. Your goal is to provide concise, actionable feedback that helps improve code quality while highlighting any critical issues.

Here's the git diff you need to review:

<git_diff>
{diff}
</git_diff>

The programming language used in this diff is:

<language>
{lang}
</language>

Instructions:
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

Output Format:
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

Example output structure:

<code_review_process>
[Your detailed analysis and observations]
</code_review_process>

## Key Issues
1. [Issue description]
   ```[language]
   [Code suggestion]
   ```

2. [Issue description]
   ```[language]
   [Code suggestion]
   ```

## Minor Improvements
- [Brief suggestion]
- [Brief suggestion]

## Ratings
- Code Quality: ⭐⭐☆
- Maintainability: ⭐⭐⭐
- Readability: ⭐⭐☆
- Performance: ⭐⭐⭐
- Security: ⭐☆☆

## Major Flags
**⚠️ [Description of critical issue, if any]**

Remember to keep your feedback concise and actionable, focusing on the most important aspects that will improve the code.`)
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
