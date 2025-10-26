from bs4 import BeautifulSoup
import csv
from io import StringIO

def extract_content_after_header(header):
    content = []
    for sibling in header.next_siblings:
        if sibling.name and sibling.name.startswith('h'):
            break
        content.append(str(sibling))
    return ' '.join(content)

# Read the content of the provided file
with open("craft_driven_research.txt", "r") as file:
    content = file.read()

# Parse the content using BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

# Extract content after the headers of interest
major_failures_header = soup.find(string="Major Australian Project Failures:")
reasons_failures_header = soup.find(string="Reason Behind such Australian Project Failures:")

if major_failures_header:
    project_failures_content = extract_content_after_header(major_failures_header.find_parent())
if reasons_failures_header:
    reasons_for_failures_content = extract_content_after_header(reasons_failures_header.find_parent())

# Extract a larger portion of the HTML content around the headers of interest for examination
start_index = content.find("Major Australian Project Failures:")
end_index = content.find("Studies by the Various Group of Researcher Companies:")

segment = content[start_index:end_index]

# Parsing the segment using BeautifulSoup
segment_soup = BeautifulSoup(segment, 'html.parser')

# Extracting project names and reasons
project_names = [li.get_text() for li in segment_soup.select("ul")[0].find_all("li")]
reasons = [li.get_text() for li in segment_soup.select("ul")[1].find_all("li")]

# Creating a CSV in-memory file
output = StringIO()
csv_writer = csv.writer(output)

# Writing to CSV
csv_writer.writerow(["Project name", "Outcome", "Reasons"])
for project in project_names:
    for reason in reasons:
        csv_writer.writerow([project, "failure", reason])

# Save the CSV content to a file
csv_filename = "projects_outcome.csv"
with open(csv_filename, "w") as csv_file:
    csv_file.write(output.getvalue())

print(f"CSV saved to: {csv_filename}")