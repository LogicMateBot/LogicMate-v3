from typing import List
from openai import OpenAI, api_key
from pydantic import BaseModel, Field, model_validator

from logicmate.models.predictions.predictions.prediction import PredictionBase
from logicmate.models.video.video import Approach, Exercise, ImageModel, Scene, Video


class OpenAIModel(BaseModel):
    """
    Class for OpenAI models.
    """

    model_for_explications: str = Field(
        default="gpt-4.1-mini", description="The name of the model for explications."
    )
    model_for_recreations: str = Field(
        default="o4-mini", description="The name of the model for recreations."
    )
    api_key: str
    client: object = Field(
        default=None,
        description="The client for OpenAI API.",
    )

    @model_validator(mode="after")
    def load_model(self) -> "OpenAIModel":
        """
        Load the OpenAI model with the specified name after initialization.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=api_key)

        if not self.client:
            raise ValueError("Client is not initialized.")

        return self

    def generate_prediction_explanation(self, prediction: PredictionBase) -> str:
        """
        Generate an explanation for a prediction using the Phi model.
        Args:
            prediction (PredictionBase): The prediction object containing the prediction data.
        Returns:
            str: The generated explanation.
        """

        base_instruction = (
            "Provide an explanation in Spanish for the given content. "
            "Do not include the class name or the original text in your response. "
            "The answer should be concise, direct, and without extra formatting or comments.\n\n"
        )

        class_instructions = {
            "code_snippet": (
                "The input is a single line of C code. Explain clearly and step by step what it does and how it works. "
                "Do not mention errors, bad practices, or suggest improvements."
            ),
            "code_bracket": (
                "The input is an opening or closing bracket ('{' or '}') in a C program. "
                "Explain its purpose in structuring code blocks."
            ),
            "initial-node": (
                "This is a start node in a flowchart. Explain what it represents and how it begins the process."
            ),
            "final-node": (
                "This is an end node in a flowchart. Explain its purpose in marking the end of the process."
            ),
            "print-node": (
                "This is a print/output node in a flowchart. Explain its role in displaying information during the process."
            ),
            "variable-node": (
                "This node represents a variable declaration or initialization. Explain its role in the algorithm."
            ),
            "operation-node": (
                "This node performs an operation. Describe what kind of operation it might be and how it affects the process."
            ),
            "decision-node": (
                "This node represents a conditional decision (like an 'if'). Explain how it influences the program flow."
            ),
            "function-node": (
                "This node represents a function call or definition. Explain its purpose within the process."
            ),
            "input-node": (
                "This node represents user input in a flowchart. Describe its importance in the logic of the algorithm."
            ),
            "output-node": (
                "This node represents data output. Explain how it is used to show results or messages to the user."
            ),
            "normal-arrow": (
                "This is a regular arrow in a flowchart. Explain how it connects steps in the process."
            ),
            "decision-arrow": (
                "This is a decision arrow (Yes/No) in a flowchart. Explain how it directs the program flow based on conditions."
            ),
        }

        specific_instruction = class_instructions.get(prediction.class_name, "")
        text_content = prediction.text or ""
        full_prompt = base_instruction + specific_instruction

        messages: list = [
            {"role": "developer", "content": full_prompt},
            {"role": "user", "content": text_content},
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_explications,
            messages=messages,
        )

        explanation = output.choices[0].message.content.strip()
        prediction.explanation = explanation
        return explanation

    def generate_image_explanation(self, image: ImageModel) -> str:
        """
        Generate an image explanation using the Phi model.

        Args:
            image (ImageModel): Image object containing associated predictions.

        Returns:
            str: The generated explanation.
        """
        if not image.predictions:
            raise ValueError(
                "Image must contain predictions to generate an explanation."
            )

        categories = image.categories or []
        if not categories:
            raise ValueError(
                "Image must have at least one category (e.g., 'code' or 'diagram')."
            )

        combined_content = "\n".join(
            [
                f"Fragment {i + 1}:\nText: {p.text or '[no text]'}\nExplanation: {p.explanation or '[no explanation]'}"
                for i, p in enumerate(image.predictions)
            ]
        )

        if "code" in categories:
            role_prompt = (
                "You are analyzing a set of C language code fragments extracted from an image. "
                "Based on the explanations and code lines, generate a concise description in Spanish "
                "explaining what the entire program or logic does. "
                "Use the explanations as support, but do not repeat them. "
                "Do not include extra commentary. Focus on summarizing what the code accomplishes as a whole."
            )
        elif "diagram" in categories:
            role_prompt = (
                "You are analyzing a set of flowchart elements extracted from an image. "
                "Based on the node texts and their explanations, generate a concise description in Spanish "
                "that summarizes the overall logic or process represented by the diagram. "
                "Use the explanations as support, but do not repeat them. "
                "Do not include extra commentary or formatting."
            )
        else:
            role_prompt = (
                "You are analyzing content that could be a mix of diagrams and code fragments. "
                "Generate a concise explanation in Spanish about what is represented, focusing on logic and purpose. "
                "Base your answer on the provided texts and explanations without repeating them directly."
            )

        messages: list = [
            {"role": "developer", "content": role_prompt},
            {
                "role": "user",
                "content": f"Contenido extraído de la imagen:\n{combined_content}",
            },
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_explications,
            messages=messages,
        )

        explanation = output.choices[0].message.content.strip()
        image.explanation = explanation
        return explanation

    def generate_scene_explanation(self, scene: Scene) -> str:
        """
        Generate a scene explanation using the Phi model based on explanations of the images within the scene.

        Args:
            scene (Scene): The scene object containing the images and their explanations.

        Returns:
            str: The generated explanation for the entire scene.
        """
        if not scene.images:
            raise ValueError("Scene must contain images.")

        if any(image.explanation is None for image in scene.images):
            raise ValueError(
                "All images must have generated explanations before explaining the scene."
            )

        scene_categories = scene.categories or []

        combined_explanations: str = "\n".join(
            [
                f"Image {i + 1}:\nExplanation: {image.explanation}"
                for i, image in enumerate(scene.images)
            ]
        )

        if "code" in scene_categories and "diagram" in scene_categories:
            role_prompt = (
                "You are analyzing a complete process represented through both C code fragments and flow diagrams. "
                "Each image has an explanation. Your goal is to summarize the overall logic and objective of the entire scene, "
                "explaining how the diagrams and code complement each other. "
                "Focus on what the full process does, not on the individual pieces. Do not repeat image explanations. "
                "Your response must be in Spanish, concise, and without additional commentary."
            )
        elif "code" in scene_categories:
            role_prompt = (
                "You are analyzing a scene composed of C language code fragments. "
                "Each image has an explanation. Based on those, describe what the full program or algorithm accomplishes. "
                "Do not repeat individual image explanations. Focus on the overall purpose and how the logic flows. "
                "Your response must be in Spanish, concise, and without additional commentary."
            )
        elif "diagram" in scene_categories:
            role_prompt = (
                "You are analyzing a scene composed of flowchart images. "
                "Each image has an explanation. Based on these, explain the general process or algorithm being represented. "
                "Do not repeat the individual image explanations. Summarize the complete logic behind the diagram sequence. "
                "Your response must be in Spanish, concise, and without extra comments or formatting."
            )
        else:
            role_prompt = (
                "You are analyzing a set of images with unknown or mixed content. "
                "Each has an explanation. Your task is to infer the overall purpose or logic of the full scene, "
                "summarizing what the combined content represents. "
                "Do not repeat the image explanations. Answer concisely and in Spanish without formatting or extra comments."
            )

        messages: list = [
            {"role": "developer", "content": role_prompt},
            {
                "role": "user",
                "content": f"Explanations extracted from the scene:\n{combined_explanations}",
            },
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_explications,
            messages=messages,
        )

        explanation = output.choices[0].message.content.strip()
        scene.explanation = explanation
        return explanation

    def generate_video_explanation(self, video: Video) -> str:
        """
        Generate a video explanation using the Phi model.

        Args:
            video (Video): The video object containing the scenes and their explanations.

        Returns:
            str: The generated explanation including a short title.
        """
        if not video.scenes:
            raise ValueError("Video must contain scenes.")

        if any(scene.explanation is None for scene in video.scenes):
            raise ValueError(
                "All scenes must have generated explanations before explaining the video."
            )

        combined_explanations: str = "\n".join(
            [
                f"Scene {i + 1}:\nExplanation: {scene.explanation}"
                for i, scene in enumerate(video.scenes)
            ]
        )

        categories = video.categories or []

        if "code" in categories and "diagram" in categories:
            role_prompt = (
                "You are analyzing a video composed of scenes that include both C code and flow diagrams. "
                "Each scene has an explanation. Based on all these, generate a concise summary in Spanish of what the video as a whole represents. "
                "Describe the overall logic or process being explained and how the code and diagrams work together. "
                "Do not repeat the scene explanations. Do not add extra commentary or formatting."
            )
        elif "code" in categories:
            role_prompt = (
                "You are analyzing a video composed of scenes that show C code explanations. "
                "Based on all scene explanations, summarize in Spanish the complete program, algorithm, or logic being presented. "
                "Avoid repeating the scene explanations and do not include additional comments."
            )
        elif "diagram" in categories:
            role_prompt = (
                "You are analyzing a video composed of scenes that contain flowchart diagrams. "
                "Using the explanations from each scene, summarize the full process or logic illustrated across the video. "
                "Keep the answer concise, do not include comments or repeat scene-level details."
            )
        else:
            role_prompt = (
                "You are analyzing a video composed of various scenes, each with its own explanation. "
                "Your task is to generate a concise summary in Spanish describing what the video overall represents or explains. "
                "Avoid extra comments or redundant explanations."
            )

        messages: list = [
            {"role": "developer", "content": role_prompt},
            {
                "role": "user",
                "content": f"Scene explanations from the video:\n{combined_explanations}",
            },
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_explications,
            messages=messages,
        )

        explanation = output.choices[0].message.content.strip()
        video.explanation = explanation
        return explanation

    def generate_video_title(self, explanation: str) -> str:
        """
        Generate a short English title for the video based on its explanation.

        Args:
            explanation (str): The explanation text of the video.

        Returns:
            str: A concise, English-language title for the video.
        """
        if not explanation:
            raise ValueError("Explanation is required to generate a title.")

        prompt: str = (
            "You are a helpful assistant that summarizes educational video explanations into short, descriptive titles. "
            "Based on the following explanation, generate a short and clear English title that accurately reflects the video's content. "
            "The title should be between 4 and 10 words, use proper capitalization, and avoid any quotation marks or punctuation at the beginning or end.\n\n"
            f"Explanation:\n{explanation}"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]

        output = self.client.chat.completions.create(
            model=self.model_for_explications,
            messages=messages,
        )

        title = output.choices[0].message.content.strip()
        return title

    def generate_code_from_video(self, video: Video) -> str:
        """
        Generate full C code from video predictions using GPT.

        Args:
            video (Video): The video object containing scenes, images, and predictions.

        Returns:
            str: The full C program generated by the model.
        """
        if not video.scenes:
            raise ValueError("The video must contain scenes.")

        code_fragments = []

        for scene in video.scenes:
            for image in scene.images:
                for prediction in image.predictions:
                    if (
                        prediction.class_name in ("code_snippet", "code_bracket")
                        and prediction.text
                    ):
                        code_fragments.append(prediction.text.strip())

        if not code_fragments:
            raise ValueError("No code-related predictions found.")

        raw_code_input = "\n".join(code_fragments)

        messages = [
            {
                "role": "developer",
                "content": (
                    "You are given a sequence of fragmented C code lines. "
                    "Your task is to reconstruct the full C program in correct order and formatting. "
                    "Structure it exactly like a real program, following this pattern:\n\n"
                    "- Include necessary libraries (e.g., stdio.h, stdlib.h, conio.h)\n"
                    "- Define macros if needed\n"
                    "- Declare function signatures\n"
                    "- Implement `main()` function\n"
                    "- Document each function using comments\n"
                    "- Add full function definitions\n\n"
                    "Use only the provided fragments and do not invent code. "
                    "Preserve variable names, logic, and formatting. "
                    "The result must be a clean, complete C program ready to compile. "
                    "Output only the code, without explanations, and in Spanish comments if any."
                ),
            },
            {
                "role": "user",
                "content": f"Here are the extracted code fragments:\n{raw_code_input}",
            },
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        final_code = output.choices[0].message.content.strip()
        video.code = final_code
        return final_code

    def generate_code_from_flowchart(self, diagram_text: str) -> str:
        """
        Generate full C code based on a flowchart.js diagram.

        Args:
            diagram_text (str): A valid flowchart.js definition.

        Returns:
            str: The generated C program.
        """
        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a C code generator. You are given a flowchart.js diagram.\n"
                    "Your task is to convert this diagram into a valid C program.\n\n"
                    "Guidelines:\n"
                    "- Interpret flowchart nodes as steps in the logic.\n"
                    "- Use meaningful indentation and structure.\n"
                    "- Include necessary headers like `stdio.h`.\n"
                    "- Use comments in Spanish to explain each block.\n"
                    "- Follow the logic as given. Don't invent new operations.\n"
                    "- Output only the final C code."
                ),
            },
            {
                "role": "user",
                "content": f"Here is the flowchart.js diagram:\n{diagram_text}",
            },
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        return output.choices[0].message.content.strip()

    def generate_flowchart_from_video(self, video: Video) -> str:
        """
        Generate a flowchart.js diagram from video predictions using GPT.

        Args:
            video (Video): The video object containing scenes, images, and predictions.

        Returns:
            str: The flowchart.js diagram generated by the model.
        """
        if not video.scenes:
            raise ValueError("The video must contain scenes.")

        all_predictions = []

        for scene in video.scenes:
            for image in scene.images:
                for prediction in image.predictions:
                    all_predictions.append(prediction.dict())

        if not all_predictions:
            raise ValueError("No diagram-related predictions found.")

        prediction_to_flowchartjs = {
            "initial-node": "start",
            "final-node": "end",
            "print-node": "output",
            "variable-node": "input",
            "operation-node": "operation",
            "decision-node": "condition",
            "function-node": "subroutine",
            "input-node": "input",
            "output-node": "output",
            "normal-arrow": "->",
        }

        base_message = {
            "role": "developer",
            "content": (
                "You are a diagram generator using the flowchart.js syntax.\n\n"
                "Your task is to convert a sequence of structured prediction nodes "
                "into a valid flowchart.js definition.\n\n"
                "Follow these strict rules:\n"
                "- Use only the supported node types in flowchart.js:\n"
                "  `start`, `end`, `operation`, `input`, `output`, `condition`, `subroutine`, `inputoutput`, `parallel`\n"
                "- Connect nodes using `->`, following their adjacency (i.e., arrows between them).\n"
                "- For condition nodes, use labeled branches like `cond(yes)->op1`, `cond(no)->op2`.\n"
                "- Do not use hyperlinks, styles, or annotations.\n"
                "- Your output must only be valid flowchart.js code, nothing else.\n"
                "- Make sure all branches lead to valid nodes, and all references are consistent.\n\n"
                "The format must be clean and renderable with flowchart.js."
            ),
        }

        node_docs = []
        for class_name, flowchart_type in prediction_to_flowchartjs.items():
            if flowchart_type == "->":
                continue
            node_docs.append(
                f"- A `{class_name}` should be translated to `{flowchart_type}` in flowchart.js.\n"
                "  Analyze its incoming and outgoing arrows to determine the control flow."
            )

        node_docs.append(
            "- A `decision-arrow` is not a node. It simply represents a labeled connection from a condition node, "
            "like `cond(yes)->step1` or `cond(no)->step2`. You must extract the label and apply it to the arrow."
        )

        node_type_message = {
            "role": "developer",
            "content": (
                "Here are the prediction class mappings you must follow:\n\n"
                + "\n".join(node_docs)
            ),
        }

        formatted_input = json.dumps(all_predictions, indent=2)
        user_message = {
            "role": "user",
            "content": f"Here are the extracted prediction nodes:\n{formatted_input}",
        }

        messages = [base_message, node_type_message, user_message]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        diagram = output.choices[0].message.content.strip()
        video.diagram = diagram
        return diagram

    def generate_flowchart_from_code(self, code_text: str) -> str:
        """
        Generate a flowchart.js diagram based on C code input.

        Args:
            code_text (str): A valid C program.

        Returns:
            str: The flowchart.js diagram representation.
        """
        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a diagram generator using the flowchart.js syntax.\n\n"
                    "You are given a valid C program. Your task is to analyze its structure and logic "
                    "and convert it into a valid flowchart.js definition that visually represents the flow of execution.\n\n"
                    "Follow these strict rules:\n"
                    "- Use only the supported node types in flowchart.js:\n"
                    "  `start`, `end`, `operation`, `input`, `output`, `condition`, `subroutine`, `inputoutput`, `parallel`\n"
                    "- Represent variable declarations, assignments, and calculations with `operation` nodes.\n"
                    "- Use `input` for `scanf` or similar input instructions.\n"
                    "- Use `output` for `printf` or similar output instructions.\n"
                    "- Use `condition` nodes to represent `if`, `if/else`, `while`, or `for` blocks.\n"
                    "- Represent reusable blocks or functions as `subroutine`.\n"
                    "- Connect nodes using `->` to show flow direction.\n"
                    "- For condition nodes, use labeled branches like `cond(yes)->step1`, `cond(no)->step2`.\n"
                    "- Every branch must connect to a valid node. Avoid floating or unreferenced nodes.\n"
                    "- Do not use hyperlinks, styles, colors, or annotations.\n"
                    "- Use only clean, syntactically valid flowchart.js code as output. No explanations, comments, or formatting.\n"
                    "- The final flowchart must be fully connected and renderable using flowchart.js."
                ),
            },
            {"role": "user", "content": f"Here is the C code:\n{code_text}"},
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        return output.choices[0].message.content.strip()

    def generate_exercises_from_code(self, video: Video) -> list[Exercise]:
        """
        Generate a list of Exercise objects based on the C code in the video.

        Args:
            video (Video): A video object with a valid C program in `video.code`.

        Returns:
            List[Exercise]: List of generated exercises.
        """
        if not video.code:
            raise ValueError("Cannot generate exercises without code.")

        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a problem generator.\n\n"
                    "You will be given a C program. Your task is to:\n"
                    "1. Understand what the program does.\n"
                    "2. Create 5-6 different exercises based on this code, each with:\n"
                    "   - A concise and clear title\n"
                    "   - A problem description (what the student should do)\n"
                    "   - The current code as the solution\n\n"
                    "Respond with a JSON array like:\n"
                    "[\n"
                    "  {\n"
                    '    "title": "...",\n'
                    '    "description": "...",\n'
                    '    "solution": "..." \n'
                    "  },\n"
                    "  ...\n"
                    "]"
                ),
            },
            {"role": "user", "content": f"Here is the C code:\n{video.code}"},
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        import json

        data = json.loads(output.choices[0].message.content.strip())

        return [Exercise(**ex) for ex in data]

    def generate_approaches_from_code(self, video: Video) -> list[Approach]:
        """
        Generate a list of Approach objects based on the C code in the video.

        Args:
            video (Video): A video object with a valid C program in `video.code`.

        Returns:
            List[Approach]: List of improved implementations and explanations.
        """
        if not video.code:
            raise ValueError("Cannot generate approaches without code.")

        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a code reviewer and optimizer.\n\n"
                    "You will be given a C program. Your task is to:\n"
                    "1. Explain how the code works (originalCodeExplanation).\n"
                    "2. Propose 5- improved versions of the code, each with:\n"
                    "   - A new version (newCode)\n"
                    "   - An explanation of why it's better (newCodeExplanation)\n\n"
                    "Respond with a JSON array like:\n"
                    "[\n"
                    "  {\n"
                    '    "title": "...",\n'
                    '    "description": "...",\n'
                    '    "originalCode": "...",\n'
                    '    "originalCodeExplanation": "...",\n'
                    '    "newCode": "...",\n'
                    '    "newCodeExplanation": "..." \n'
                    "  },\n"
                    "  ...\n"
                    "]"
                ),
            },
            {"role": "user", "content": f"Here is the C code:\n{video.code}"},
        ]

        output = self.client.chat.completions.create(
            model=self.model_for_recreations,
            messages=messages,
        )

        import json

        data = json.loads(output.choices[0].message.content.strip())

        return [Approach(**ap) for ap in data]

    def generate_explanation(self, video: Video) -> Video:
        """
        Generate a complete explanation and produce both code and flowchart,
        depending on the video category ('code' or 'diagram').

        Args:
            video (Video): The video object containing the video data.

        Returns:
            Video: The same video object with explanations and generated content.
        """

        for scene in video.scenes:
            for image in scene.images:
                if not image.predictions:
                    continue
                for prediction in image.predictions:
                    prediction.explanation = self.generate_prediction_explanation(
                        prediction=prediction
                    )
                image.explanation = self.generate_image_explanation(image=image)
            scene.explanation = self.generate_scene_explanation(scene=scene)

        video.explanation = self.generate_video_explanation(video=video)
        video.title = self.generate_video_title(explanation=video.explanation)

        categories: List[str] = video.categories or []

        if "code" in categories and not video.code:
            video.code = self.generate_code_from_video(video=video)

        if "diagram" in categories and not video.diagram:
            video.diagram = self.generate_flowchart_from_video(video=video)

        if video.code and not video.diagram:
            video.diagram = self.generate_flowchart_from_code(code_text=video.code)

        if video.diagram and not video.code:
            video.code = self.generate_code_from_flowchart(diagram_text=video.diagram)

        if video.code:
            video.exercises = self.generate_exercises_from_code(video=video)
            video.approaches = self.generate_approaches_from_code(video=video)

        return video
