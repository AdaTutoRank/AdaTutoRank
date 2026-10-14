from typing import Dict, List, Optional

class Question:
    """
    Represents a question with automatic validation.

    Attributes:
        question (str): The text of the question.
    """
    def __init__(self, question: str ) -> None:
        """
        Initializes a Question instance.

        Args:
            question (str): The text of the question.

        Example:
            ```python
            q = Question("What is the capital of France?")
            print(q)  # Output: Question: What is the capital of France?
            ```
        """
        self.question = self.check_question(question)
    @classmethod
    def check_question(cls,question) -> str:
        """
        Ensures the question ends with a question mark.

        Args:
            question (str): The text of the question.

        Returns:
            str: The question with a question mark at the end if it was missing.

        Example:
            ```python
            Question.check_question("What is the capital of France")
            # Output: 'What is the capital of France?'
            ```
        """
        cls.question = question if question.endswith("?") else question +"?"
        return cls.question
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Question instance.

        Returns:
            str: The formatted question.

        Example:
            ```python
            q = Question("What is the capital of France?")
            str(q)  # Output: 'Question: What is the capital of France?'
            ```
        """
        return f"Question: {self.question}"
    
class Answer:
    """
    Represents answers to a question.

    Attributes:
        answers (list[str]): A list of possible answers.
    """
    def __init__(self, answers:list=None) -> None:
        """
        Initializes an Answer instance.

        Args:
            answers (list[str] or str, optional): A list of possible answers. Defaults to None.

        Example:
            ```python
            a = Answer(["Paris", "Lyon"])
            print(a)  
            # Output:
            # Answer:
            # - Paris
            # - Lyon
            ```
        """
        if isinstance(answers, str):  # If it's a string, convert it to a list
            self.answers = [answers]
        elif isinstance(answers, int):
             self.answers = [str(answers)]
        elif isinstance(answers, list):  # If it's a list, ensure all elements are strings
            self.answers = [str(answer) for answer in answers]
        else:  # If it's neither, initialize with an empty list
            self.answers = []

    def __str__(self) -> str:
        """
        Returns a string representation of the Answer instance.

        Returns:
            str: The formatted answers.

        Example:
            ```python
            a = Answer(["Paris", "Lyon"])
            str(a)  
            # Output: 
            # Answer: 
            # - Paris
            # - Lyon
            ```
        """
        return f"Answer: \n- "+ "\n- ".join(self.answers)

class Context:
    """
    Represents a context with metadata such as score and title.

    Attributes:
        score (float, optional): The relevance score of the context.
        has_answer (bool, optional): Whether the context contains an answer.
        id (int, optional): The identifier of the context.
        title (str, optional): The title of the context.
        text (str, optional): The text of the context.
    """
    def __init__(self, score: float=None, has_answer: bool=None, id: str=None, title: str=None, text: str=None)-> None:
        """
        Initializes a Context instance.

        Args:
            score (float, optional): The relevance score.
            has_answer (bool, optional): Whether the context contains an answer.
            id (int, optional): The identifier of the context.
            title (str, optional): The title of the context.
            text (str, optional): The text of the context.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(c)
            ```
        """
        self.score: Optional[float] = score
        self.has_answer: Optional[bool] = has_answer
        self.id: Optional[str] = id
        self.title: Optional[str] = title
        self.text: Optional[str] = text

    def to_dict(self, save_text: bool=False) -> Dict[str, Optional[object]]:

        """
        Converts the Context instance to a dictionary.

        Args:
            save_text (bool): Whether to include text in the output dictionary.

        Returns:
            dict: The context data.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(c.to_dict())
            ```
        """
        context_dict = {
            "score": float(self.score) if self.score is not None else None,
            "has_answer": self.has_answer,
            "id": self.id,
            }

        # Include 'text' only if save_text is True
        if save_text:
            context_dict["text"] = self.text
            context_dict["title"] =  self.title

        return context_dict
    def __str__(self) -> str:
        """
        Returns a string representation of the Context instance.

        Returns:
            str: The formatted context.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(str(c))
            ```
        """
        return f"ID: {self.id}\nHas Answer: {self.has_answer}\nTitle: {self.title}\nText: {self.text}\nScore: {self.score}"

class Document:
    """
    Represents a document consisting of a question, answers, and contexts.

    Attributes:
        question (Question): The question associated with the document.
        answers (Answer): The answers to the question.
        contexts (list[Context]): A list of related contexts.
        reorder_contexts (list[Context] or None): A reordered list of contexts based on relevance.
    """
    def __init__(self, question: Question, answers: Answer, contexts: list = None , id: int = None) -> None:
        """
        Initializes a Document instance.

        Args:
            question (Question): The question associated with the document.
            answers (Answer): The answers to the question.
            contexts (list[Context], optional): A list of contexts related to the question.

        Example:
            ```python
            q = Question("What is the capital of France?")
            a = Answer(["Paris"])
            c1 = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            c2 = Context(score=0.5, has_answer=False, id=2, title="Berlin", text="Berlin is the capital of Germany.")
            d = Document(question=q, answers=a, contexts=[c1, c2])
            print(d)
            ```
        """
        self.question: Question = question
        self.answers: Answer = answers
        self.contexts: List[Context] = contexts
        self.reorder_contexts: List[Context] = None
        self.id = str(id) 

    @classmethod
    def from_dict(cls, data: dict,n_docs:int=100) -> 'Document':
        """
        Creates a Document instance from a dictionary.

        Args:
            data (dict): A dictionary containing the question, answers, and contexts.
            n_docs (int, optional): The number of contexts to include. Defaults to 100.

        Returns:
            Document: A new Document instance.

        Example:
            ```python
            data = {
                "question": "What is the capital of France?",
                "answers": ["Paris"],
                "ctxs": [
                    {"score": 0.9, "has_answer": True, "id": 1, "title": "Paris", "text": "The capital of France is Paris."},
                    {"score": 0.5, "has_answer": False, "id": 2, "title": "Berlin", "text": "Berlin is the capital of Germany."}
                ]
            }
            d = Document.from_dict(data)
            print(d.question)
            ```
        """
        question = Question(data["question"])
        if "answers" in data:
            answers = Answer(data["answers"])
        else:
            answers =Answer('')
        
        if "query_id" in data:
            id = data["query_id"]
        else:
            id = None
        contexts = [Context(**ctx) for ctx in data["ctxs"][:n_docs]]
        return cls(question, answers, contexts, id=id)

    def to_dict(self) -> Dict[str, Optional[object]]:
        """
        Converts the document into a dictionary representation.

        Returns:
            dict: A dictionary containing the question, answers, and contexts.
        """
        return {
            "question": self.question.question,
            "answers": self.answers.answers,
            "contexts": [ctx.to_dict() for ctx in self.contexts]
        }
    def to_dict_reoreder(self) -> Dict[str,Optional[object]]:
        return {
            "question" : self.question.question,
            "answers" : self.answers.answers,
            "contexts" : [ctx.to_dict() for ctx in self.reorder_contexts]
        }
    def __str__(self) -> str:
        """
        Returns a string representation of the Document instance.

        Returns:
            str: The formatted document information.

        Example:
            ```python
            d = Document(Question("What is the capital of France?"), Answer(["Paris"]))
            print(d)
            ```
        """
        contexts_str = "\n\n".join([str(ctx) for ctx in self.contexts])
        reorder_contexts_str= ''
        if self.reorder_contexts is not None:
            reorder_contexts_str = "\n\n".join([str(ctx) for ctx in self.reorder_contexts])
        return f"{self.question}\n\n{self.answers}\n\nContext: \n\n{contexts_str}\nReorder contexts: \n\n{reorder_contexts_str}"
