from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(tool.invoke("F1 movie release date"))
