import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['SERPER_API_KEY'] = SERPER_API_KEY

search_tool = SerperDevTool()
llm = ChatGroq(model_name= "llama3-8b-8192")

def create_research_agents():
    return Agent(
        role = "Research Specialist",
        goal = "Conduct through research on given topics",
        backstory = "you are an Experiend researcher with expertise in finding and synthesizing from various resources",
        verbose = True,
        allow_deligation = False,
        tools =[search_tool],
        llm = llm
        

        
        
        
    )

def create_research_task(agent, topic):
    return Task(
        description = f"Research the following topic and provide a comprehensive summery:{topic}",
        agent = agent,
        expected_output = "A detailed summary of the research findings, including keypoints and insights"
        
        
    )
    
    
def run_research(topic):
        agent= create_research_agents()
        task = create_research_task(agent, topic)
        crew= Crew(agent=[agent], tasks=[task])
        result = crew.kickoff()
        return result
        
        
if __name__ == "__main__":
    print("Welcome to the Agents")
    topic = input("enter the research topic: ")
    result = run_research(topic)
    print("Research results: ")
    print(result)