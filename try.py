from conns import gemini_client

res = gemini_client.models.generate_content_stream(contents=["hi"], model='gemini-2.0-flash')
for chunk in res:
    print(chunk)