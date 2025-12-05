import dotenv

dotenv.load_dotenv()
import os

from openai import AzureOpenAI


class LLM():
    """
    This class provides an API to obtain LLM outputs.
    """

    def __init__(self):
        """
        Sets up an instance of the LLM class by configuring the LLM client.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        self.openai_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-06-01",
            azure_endpoint="https://hkust.azure-api.net",
            azure_deployment="gpt-4o-mini",
        )

    
    def _get_sys_prompt(self) -> str:
        """
        Provides an easy way to obtain the system prompt using methods in this class.

        returns:
        - a string containing the system prompt to use the LLM as a palmistry expert.
        """
        system_prompt = """You are a palmistry expert. 
You will be presented with an image of a palm.
Please perform analysis on the image of the palm as instructed.

## Supplementary Information on Palmistry
Palm lines of interest include the *Life Line*, the *Heart Line*, the *Fate Line*, and the *Head Line*.
These lines represent respectively, *enthusiasm and strength*, *romantic life*, *fortune and luck*, and *smartness and potential*.
You are to read these lines in detail, and to provide insights in the form of scores.

### Heart Line
- Begins below the index finger = content with love life
- Begins below the middle finger = selfish when it comes to love
- Begins in-between the middle and index fingers = caring and understanding
- Is straight and short = less interest in romance
- Touches life line = heart is broken easily
- Is long and curvy = freely expresses emotions and feelings
- Is straight and parallel to the head line = good handle on emotions
- Is wavy = many relationships, absence of serious relationships
- Circle on the line = sad or depressed
- Broken line = emotional trauma

### Head Line
- Short line = prefers physical achievements over mental ones
- Curved, sloping line = creativity
- Curves downward = inclination towards literature and fantasy
- Curves upwards towards little finger = aptitude for math, business, and logic
- Separated from life line = adventure, enthusiasm for life
- Wavy line = short attention span
- Deep, long line = thinking is clear and focused
- Straight line = thinks realistically
- Broken head line = inconsistencies in thought or has varying interests
- Multiple crosses through head line = momentous decisions

### Life Line
- Runs close to thumb = often tired
- Curves completely around the thumb = good physical and mental health
- Forked upwards = positive attitude towards life
- Forked downwards = pessimist
- Curvy = plenty of energy
- Forms a semicircle = enthusiastic and courageous
- Long and deep = vitality
- Short and shallow = manipulated by others
- Swoops around in a semicircle = strength and enthusiasm
- Straight and close to the edge of the palm = cautious when it comes to relationships
- Ends at base of index finger = academic achievement
- Ends at base of pinky finger = success in business
- Ends at base of ring finger = sign of wealth
- Ends below the thumb = strong attachment with family
- Multiple life lines = extra vitality
- Circle in line = hospitalized or injured
- Break in line = sudden change of lifestyle
- No line = nervous

### Fate Line
- Deep line = strongly controlled by fate
- Unbroken and runs straight across = successful life ahead
- Breaks and changes of direction = prone to many changes in life from external forces
- Fork in the line = great amount of wealth ahead
- Starts joined to life line = self-made individual; develops aspirations early on
- Joins with life line somewhere in the middle = signifies a point at which one’s interests must be surrendered to those of others
- Starts at base of thumb and crosses life line = support offered by family and friends
- No line = comfortable but uneventful life ahead

## Scoring Instructions
Provide scores for the user for `strength`, `romantic`, `luck`, and `potential`.
These correspond to the above points that you were asked to pay attention to previously.
Your scores should be within 0 and 1, with 1 being the highest possible score.
Give your output in the form of a JSON string, with the score keys being `strength`, `romantic`, `luck` and `potential`.
Do *NOT* place ANY markdown backticks in your output, as the output will be directly parsed in a Python script.
        """
        return system_prompt


    def get_LLM_output(
            self, 
            user_prompt: str, 
            system_prompt: str = None,
            image_data: str = None
        ) -> str:
        """
        Gets output from an LLM.

        args:
        - image_data (str): image URL after being encoded
        - user_prompt (str): user query towards the LLM
        - system_prompt (str): system prompt that provides context and instructions to the LLM

        returns:
        - a string that contains the LLM's output
        """
        sys_prompt = system_prompt if system_prompt else self._get_sys_prompt()
        
        messages = [
            {"role": "system", "content": sys_prompt}
        ]

        if image_data:
            user_message = [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            ]
        else:
            user_message = user_prompt

        messages.append({"role": "user", "content": user_message})

        output = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        return output.choices[0].message.content