import openai

openai.api_key = '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'

def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True

# Check the validity of the API key
api_key_valid = is_api_key_valid()
print("API key is valid:", api_key_valid)