from fastapi import FastAPI, Depends
import uvicorn

app = FastAPI()

def get_dep():
    return "dependency_value"

def factory(name):
    def handler(dep = Depends(get_dep)):
        print(f"Handler called with dep: {dep}")
        return f"{name}: {dep}"
    return handler

# Syntax 1
app.get("/test1")(factory("test1"))

# Syntax 2
f2 = factory("test2")
app.get("/test2")(f2)

if __name__ == "__main__":
    # Just checking if definition crashes
    print("Defined routes")
