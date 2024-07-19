from langchain_community.llms.llamafile import Llamafile

class Generator:
    def __init__(self, model_url: str, n_predict: int=400, temperature: float=0.3):
        self.generator = Llamafile(base_url=model_url, n_predict=n_predict, temperature=temperature)

    def generate_text(self, prompt: str) -> str:
        """
            Function for text generation based on the used prompt. 
        """
        text = self.generator.invoke(prompt)
        return text
