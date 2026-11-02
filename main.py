from dotenv import load_dotenv;
from langchain_core.prompts import PromptTemplate;
from langchain_google_genai import ChatGoogleGenerativeAI;
from langchain_ollama import ChatOllama;
import os;

load_dotenv();


print("API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

def main():
    print("Hello from llm-learning!");
    template = """Los Angeles (often shortened as LA or L.A.)[b] is the most populous city in the U.S. state of California, and the commercial, financial, and cultural center of Southern California. With an estimated 3.87 million residents within the city limits as of 2025,[8] it is the second-most populous city in the United States, behind New York City, and the largest city in the Western United States. The city has an ethnically and culturally diverse population, and is the principal city of a metropolitan area of 12.9 million people (2024). Greater Los Angeles, a combined statistical area that includes the Los Angeles and Riverside–San Bernardino metropolitan areas, is a sprawling metropolis of over 18 million residents.

The majority of the city proper lies in the Los Angeles Basin adjacent to the Pacific Ocean in the west and extending partly through the Santa Monica Mountains and north into the San Fernando Valley, with the city bordering the San Gabriel Valley to its east. It covers about 469 square miles (1,210 km2),[6] and is the county seat and most populated city of Los Angeles County, which is the most populous county in the United States with an estimated 9.86 million residents as of 2022.[15] It is the third-most visited city in the U.S. (after New York City and Miami), with over 3.6 million visitors as of 2023.[16]

The area that became Los Angeles was originally inhabited by the indigenous Tongva people and later claimed by Juan Rodríguez Cabrillo for Spain in 1542. The city was founded on September 4, 1781, under Spanish governor Felipe de Neve, on the village of Yaanga.[17] It became a part of the First Mexican Empire in 1821 following the Mexican War of Independence. In 1848, at the end of the Mexican–American War, Los Angeles and the rest of California were purchased as part of the Treaty of Guadalupe Hidalgo and became part of the United States. Los Angeles was incorporated as a municipality on April 4, 1850, five months before California achieved statehood. The discovery of oil in the 1890s brought rapid growth to the city.[18] The city was further expanded with the completion of the Los Angeles Aqueduct in 1913, which delivers water from Eastern California.

Los Angeles has a diverse economy with a broad range of industries. Despite a decline in film production, it is the oldest and one of the largest hubs of film and television production.[19] It has one of the busiest container ports in the Americas.[20][21][22] Despite an exodus of commerce and an increasingly unreliable water supply, Downtown Los Angeles is evolving as a cultural center with the world's largest showcase of architecture designed by Frank Gehry.[23] In 2024, the Los Angeles metropolitan area had a gross metropolitan product of over $1.295 trillion,[24] making it the city with the third-largest GDP in the world, after New York and Tokyo. Los Angeles hosted the Summer Olympics in 1932 and 1984, and will host the games for a third time in 2028."""

    summary_template = """
Give summary about {place} about this place:
1. Summarize the following text in 3-4 sentences.
2. List 2 important points about the place.
"""

    summary_prompt = PromptTemplate(
    input_variables=["place"],
    template=summary_template,
)

    llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_output_tokens=1000
)
#     llm = ChatOllama(
#     model="gemma3:270m",
#     temperature=0.7
# )
    chain = summary_prompt | llm
    result = chain.invoke({
    "place": template
})
    print(result.text)

if __name__ == "__main__":
    main();