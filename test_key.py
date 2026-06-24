import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)
print(msg.content[0].text)
