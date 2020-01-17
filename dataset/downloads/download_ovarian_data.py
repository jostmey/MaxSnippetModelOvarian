#
# This script is VDJServer specific.
# This script is specific to this dataset.
#
# We do not want to commit large data files into the
# github repository, so we download from VDJServer
# Community Data Portal.
#

import requests
import os
vdjserver_api='https://vdjserver.org/api/v1'

# The VDJServer public project uuid
project='3276777473314001386-242ac116-0001-012'

# The igblast job and archive file
tcell_file='8126808338152821226-242ac116-0001-012'
tcell_job='262bfb78-4758-4d0d-819c-49f9661d69ed-007'

# The adaptive biotech files
ab_file='2281015004902256150-242ac116-0001-012'

def download_file(project_uuid, file_uuid, filename):
    url = vdjserver_api + '/projects/' + project_uuid + '/postit/' + file_uuid
    resp = requests.get(url)
    resp.raise_for_status()
    obj = resp.json()
    href = obj['result']['_links']['self']['href']
    with requests.get(href, stream=True) as r:
        r.raise_for_status()
        print("Downloading: " + href)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return filename

# T-cell data
# generate posit then download data
download_file(project, ab_file, './data.zip')
os.system("unzip ./data.zip")
os.system("rm -f ./data.zip")
#os.system("cd " + tcell_job + " && ls *.airr.tsv.zip | xargs -n 1 unzip")
