from locust import HttpUser, task, between
import json
import random
import time


class PerformanceTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def testApi(self):
        inputs = ["i love you to the moon and back",
                  "i hate you",
                  "you are bad guy but i still love you"]

        self.client.post("/predictions/sentiment", json={"input": random.choice(inputs)})
