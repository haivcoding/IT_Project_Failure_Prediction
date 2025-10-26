import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_project_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract project details based on the HTML structure of the page
    # This may need to be adjusted based on the actual structure of the webpage
    project_details = {
        'Project name': soup.find('h1').text if soup.find('h1') else 'Not specified',
        # ... add other details extraction here
    }
    return project_details

def main():
    base_url = 'https://calleam.com/WTPF/?page_id=3'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    projects = []
    for link in soup.find_all('a', href=True):
        if 'project' in link['href']:  # adjust this condition based on the actual URLs of the project pages
            project_url = link['href']
            project_details = get_project_details(project_url)
            projects.append(project_details)
    
    df = pd.DataFrame(projects)
    df.to_csv('failed_projects.csv', index=False)

if __name__ == '__main__':
    main()