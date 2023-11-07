import os
from os import environ
from Bard import Chatbot

os.environ["BARD__Secure_1PSID"] = "cwjFA9Hl1R4lGWMyS8eTL1ztQNoEU1-fLm53uF6WGolyLwdp7iCi3rrYpwmAjB4OFpUSvQ."
os.environ["BARD__Secure_1PSIDTS"] = "sidts-CjIBNiGH7qqLZKPU2Hr6AHUXOaCDllIqEDIsIByf_SBCeJFBpgaraqyqhqOzcqXTHaNHMBAA"

Secure_1PSID = environ.get("BARD__Secure_1PSID")
Secure_1PSIDTS = environ.get("BARD__Secure_1PSIDTS")
chatbot = Chatbot(Secure_1PSID, Secure_1PSIDTS)

response = chatbot.ask("Hello, how are you?")

print(response)