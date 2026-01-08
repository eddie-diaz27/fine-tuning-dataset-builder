# Custom Mode Custom Prompts

## Purpose
Custom Mode allows you to define completely custom behavior for dataset generation.

## How to Use

### Option 1: File-Based Prompt
1. Create a file named `prompt.txt` in this directory
2. Write your custom instructions
3. Set `agent.custom.use_prompt_file: true` in `config.yaml`
4. Run the system

### Option 2: Interactive Prompt
1. Set `agent.custom.use_prompt_file: false` in `config.yaml`
2. Run the system
3. You'll be prompted to enter instructions interactively
4. Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done

## Prompt Template

Your prompt should include:

```
OBJECTIVE:
<What should the dataset accomplish?>

CONTEXT:
<What domain/topic/use case?>

FORMAT:
<How should conversations be structured?>

QUALITY CRITERIA:
<What makes a good example?>

SPECIAL INSTRUCTIONS:
<Any unique requirements?>
```

## Example Prompt

```
OBJECTIVE:
Generate a dataset for fine-tuning a Python programming tutor.

CONTEXT:
Focus on intermediate Python concepts: decorators, generators, context managers, async/await.

FORMAT:
SYSTEM: You are an experienced Python developer and teacher.
USER: <student question>
ASSISTANT: <clear, pedagogical explanation with code examples>

QUALITY CRITERIA:
- Code examples must be syntactically correct
- Explanations should be clear but not overly simplistic
- Include both "why" and "how"
- Mention common pitfalls

SPECIAL INSTRUCTIONS:
- Vary difficulty within intermediate range
- Include at least one code example per exchange
- Occasionally include multi-turn conversations where student asks follow-up questions
```

## Advanced Features

If you set `agent.custom.allow_advanced: true` in config.yaml, you can:
- Define multi-stage generation workflows
- Use experimental techniques
- Combine extraction with custom generation logic
- Implement domain-specific validation

## Tips

1. **Be Specific**: Vague prompts lead to mediocre results
2. **Provide Examples**: Show the format you want
3. **Set Quality Bar**: Define what "good" looks like
4. **Iterate**: Test with small dataset first, refine prompt, then scale up
5. **Document**: Keep notes on what works for future reference
