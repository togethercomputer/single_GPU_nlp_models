import asyncio
import os
from typing import Dict
import json
import nats
import traceback


class FastInferenceInterface:
    def __init__(self, model_name: str, args=None) -> None:
        self.model_name = model_name

    def infer(self, job_id, args) -> Dict:
        pass

    async def on_message(self, msg):
        instruction = json.loads(msg.data.decode("utf-8"))
        instruction['args'] = json.loads(instruction['args'])
        try:
            if isinstance(instruction['prompt'], list):
                instruction['args']['prompt'] = instruction['prompt']
            elif isinstance(instruction['prompt'], str):
                instruction['args']['prompt'] = [instruction['prompt']]
            else:
                raise TypeError("Only str or list of str are allowed")
        except Exception as e:
            traceback.print_exc()
            print("error in inference: "+str(e))

        # instruction['args']['temperature'] = instruction.get('temperature', 0.9)
        # instruction['args']['top_p'] = instruction.get('top_p', 0)
        # instruction['args']['max_tokens'] = instruction.get('max_tokens', 16)
        # instruction['args']['stop'] = instruction.get('stop', [])
        # instruction['args']['echo'] = instruction.get('echo', False)
        if 'temperature' not in instruction['args']:
            instruction['args']['temperature'] = 0.9
        if 'top_p'not in instruction['args']:
            instruction['args']['top_p'] = 0
        if 'max_tokens' not in instruction['args']:
            instruction['args']['max_tokens'] = 16
        if 'stop' not in instruction['args']:
            instruction['args']['stop'] = []
        if 'echo' not in instruction['args']:
            instruction['args']['echo'] = False

        instruction['args']['seed'] = instruction.get('seed', 3406)
        job_id = instruction['id']
        if isinstance(job_id, str):
            job_id = [job_id]
        try: 
            self.infer(job_id, instruction['args'])
        except Exception as e:
            traceback.print_exc()
            print("error in inference: "+str(e))

    def on_error(self, ws, msg):
        print(msg)

    def on_open(self, ws):
        ws.send(f"JOIN:{self.model_name}")

    def start(self):
        nats_url = os.environ.get("NATS_URL", "localhost:8092/my_coord")
        async def listen():
            nc = await nats.connect(f"nats://{nats_url}")
            sub = await nc.subscribe(subject=self.model_name, queue=self.model_name, cb=self.on_message)
        loop = asyncio.get_event_loop()
        future = asyncio.Future()
        asyncio.ensure_future(listen())
        loop.run_forever()


if __name__ == "__main__":
    fip = FastInferenceInterface(model_name="gpt-j-6b")
    fip.start()
