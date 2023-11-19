from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simplet5 import SimpleT5

model = SimpleT5()
# model.from_pretrained(model_type="t5", model_name="t5-base")

model.load_model("t5","/home/adarsh/IMD/backend/simplet5-epoch-6-train-loss-1.0673-val-loss-0.5636/", use_gpu=False)


app = FastAPI()

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)



@app.get("/echo/{text}")
async def echo_text(text: str):
  print(text)
  t = model.predict("answer: " + text)
  return f"{t[0]}"

