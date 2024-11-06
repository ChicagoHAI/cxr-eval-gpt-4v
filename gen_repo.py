# for prompt 1.1 1.2 1.3 1.5
#%%
import base64
from mimetypes import guess_type
import requests
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from openai import AzureOpenAI
from io import BytesIO
from PIL import Image
import os

def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, dest='api_key', 
                        help="API key for Azure API.")
    parser.add_argument("--endpoint", type=str, dest='endpoint', 
                        help="Endpoint for Azure API.")
    parser.add_argument("--m", type=str, 
                        dest='image_dir', default=None, 
                        help="Directory path to image files.")
    parser.add_argument("--i", type=str, 
                        dest="input_dir", default=None, 
                        help="Directory path to input csv(s).")
    parser.add_argument("--o", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output csv(s).")
    parser.add_argument("--p", type=str, 
                        dest="prompt_dir", default=None, 
                        help="Directory path to prompts.")
    parser.add_argument("--type", type=str, default='1', 
                        help="Choose the type of prompts you want to input.")
    args = parser.parse_known_args()

    return args


# (In) gt_indc_sample.csv + ind_list_sample.csv 
# (Out) gen_findings_sample.csv + gen_imp_sample.csv + err_list.txt : study_id (str), report (str)
class GenRepoAzureGPT:
    def __init__(self, api_key, endpoint, image_dir, input_dir, output_dir, prompt_dir, max_retries=3, prompt='1'):
        self.api_key = api_key
        self.endpoint = endpoint
        self.prompt_label = prompt
        self.image_dir = image_dir
        self.in_ind_pth = input_dir + 'ind_list_sample.csv'
        self.in_indc_pth = input_dir + 'gt_indc_sample.csv'
        self.out_dir = output_dir
        self.out_findings_pth = output_dir + 'gen_findings_sample_{}.csv'.format(self.prompt_label)
        self.out_imp_pth = output_dir + 'gen_imp_sample_{}.csv'.format(self.prompt_label)
        self.max_retries = max_retries
        self.prompt_dir = prompt_dir



    def encode_image(self, image_path):
        mime_type, _ = guess_type(self.image_dir+image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(self.image_dir+image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"


    def get_few_shot_prompt(self):
        df_shot_img = pd.read_csv(self.prompt_dir+'5_shot_img.csv')
        df_shot_img = df_shot_img[df_shot_img['dataset']=='openi'].reset_index(drop=True) ### can also modify to 'mimic' / 'openi' / 'chexpert'
        df_shot_repo = pd.read_csv(self.prompt_dir+'5_shot_repo.csv')
        df_shot_repo = df_shot_repo[df_shot_repo['dataset']=='openi'].reset_index(drop=True) ### can also modify to 'mimic' / 'openi' / 'chexpert'

        # encode image
        ls_base64 = []
        for i in range(df_shot_img.shape[0]):
            pth = df_shot_img['img_pth'][i]
            base64_image = self.encode_image(pth)
            ls_base64.append(base64_image)
        df_shot_img['img_url'] = ls_base64

        # prompt
        ls_5_shot = []
        for i in range(df_shot_repo.shape[0]):
            one_shot = []
            ls_base64_one_shot = list(df_shot_img[df_shot_img['study_id']==df_shot_repo['study_id'][i]]['img_url'])
            for url in ls_base64_one_shot:
                one_shot.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                    }
                )
            one_shot.append(
            {
                "type": "text",
                "text": f"FINDINGS: {df_shot_repo[df_shot_repo['study_id']==df_shot_repo['study_id'][i]]['findings'].iloc[0]}"
            }
            )
            one_shot.append(
            {
                "type": "text",
                "text": f"IMPRESSION: {df_shot_repo[df_shot_repo['study_id']==df_shot_repo['study_id'][i]]['impression'].iloc[0]}"
            }
            )
            ls_5_shot.append(one_shot)

        return ls_5_shot
  
    def get_usr_prompt(self, ls_base64_image, indc):
        # user text prompt
        with open(os.path.join(self.prompt_dir, self.prompt_label+'.txt'), 'r') as file:
            prompt = file.read()

        ls_text = [
            {
                    "type": "text",
                    "text": prompt.format(indc)
                }
        ]

        # user image prompt
        ls_img = []
        for i in range(len(ls_base64_image)):
            ls_img.append(
                {
                    "type": "image_url",
                    "image_url": {
                    "url": ls_base64_image[i]
                    }
                }
            )

        ls_content = ls_text + ls_img
        return ls_content, ls_text, ls_img


    def gen_per_request(self, ls_base64_image, indc, api_version = '2024-02-01', deployment_name = 'gpt-4o-test'): ### api_version: 2023-12-01-preview, deployment_name=gpt-4v
        '''
        Report Generation for each request.
        '''
        client = AzureOpenAI(
            api_key=self.api_key,  
            api_version=api_version,
            base_url=f"{self.endpoint}/openai/deployments/{deployment_name}"
        )

        # user prompt
        ls_content_usr, ls_text_usr, ls_img_usr = self.get_usr_prompt(ls_base64_image, indc)

        # system + user prompt
        if self.prompt_label == '1_5':
            # 5-shot prompt
            ls_few_shot = self.get_few_shot_prompt()

            ls_message = [
                {
                "role": "system",
                "content": "You are a professional chest radiologist that reads chest X-ray image."
                },
                {
                "role": "user",
                "content": ls_text_usr
                }
            ]
            for i in range(len(ls_few_shot)):
                ls_message.append(
                {
                    "role": "user",
                    "content": ls_few_shot[i]
                }
                )
            ls_message.append(
                {
                "role": "user",
                "content": ls_img_usr
                }
            )
        else:
            ls_message = [
                {
                "role": "system", 
                "content": "You are a professional chest radiologist that reads chest X-ray image."
                },
                {
                "role": "user",
                "content": ls_content_usr
                }
            ]

        # text+image prompt
        apiresponse = client.chat.completions.with_raw_response.create(
            model=deployment_name,
            messages=ls_message,
            max_tokens=4096
        )

        debug_sent = apiresponse.http_request.content
        chat_completion = apiresponse.parse()
        
        return chat_completion



    def gen_per_study(self, id, df_indc_s, df_ind_s, dict_ind_error):
        '''
        Report Generation for each study.
        '''
        ls_pth_img = list(df_ind_s[df_ind_s['study_id'] == id]['path_image'])

        # Prepare encoded images
        ls_base64_image = []
        for pth in ls_pth_img:
            base64_image = self.encode_image(pth)
            ls_base64_image.append(base64_image)

        # Generate reports with max_retries = 3
        # Total number of images should not exceed 10.
        temp_repo = ''
        for n in range(self.max_retries):
            temp_resp = self.gen_per_request(ls_base64_image, df_indc_s[df_indc_s['study_id'] == id]['report']) 
            try:
                temp_repo = temp_resp.choices[0].message.content
            except KeyError as e:
                if n == self.max_retries-1:
                    print(f's{id} still encounters request rejection after {n+1} tries:', e)
                    dict_ind_error[id] = temp_resp
            
            if "IMPRESSION" in temp_repo:
                break

        # Save temp results and update result dataframe for each request
        if not os.path.exists(self.out_dir + f'cache/{self.prompt_label}'):
            os.makedirs(self.out_dir + f'cache/{self.prompt_label}', exist_ok=True)

        with open(self.out_dir + f'cache/{self.prompt_label}/s{id}.txt', 'w') as file:
            file.write(temp_repo)

        return n, temp_repo, dict_ind_error


    def repo_split(self, repo_str, id, dict_ind_error, df_gen_findings_s, df_gen_imp_s):
        '''
        For this function, we require that FINDINGS is before IMPRESSION.
        '''
        df_gen_findings = pd.DataFrame(columns=['study_id', 'report'], index=[0])
        df_gen_imp = pd.DataFrame(columns=['study_id', 'report'], index=[0])

        try:
            if "FINDINGS:" in repo_str:
                temp_repo = repo_str.split("FINDINGS:")[1].split("IMPRESSION:")
                repo_imp = temp_repo[1].strip()
                cleaned_imp = " ".join(repo_imp.split())
                repo_findings = temp_repo[0].strip()
                cleaned_findings = " ".join(repo_findings.split())
            elif "IMPRESSION:" in repo_str:
                temp_repo = repo_str.split("IMPRESSION:")[1]
                repo_imp = temp_repo.strip()
                cleaned_imp = " ".join(repo_imp.split())
                cleaned_findings = np.nan
            else:
                cleaned_findings = np.nan
                cleaned_imp = np.nan 
        except IndexError as i:
            print(f's{id} encounters format error:', i)
            dict_ind_error[id] = i
            cleaned_findings = np.nan
            cleaned_imp = np.nan

        df_gen_findings['study_id'] = id
        df_gen_findings['report'] = cleaned_findings
        df_gen_imp['study_id'] = id
        df_gen_imp['report'] = cleaned_imp

        df_gen_findings_s = pd.concat([df_gen_findings_s, df_gen_findings])
        df_gen_imp_s = pd.concat([df_gen_imp_s, df_gen_imp])

        return df_gen_findings_s, df_gen_imp_s, dict_ind_error

    def run(self):
        """
        - ind_list_sample.csv  must have column 'study_id', 'path_image'.
        """
        # read data files
        df_ind_s = pd.read_csv(self.in_ind_pth).sort_values(by='study_id').reset_index(drop=True)
        ls_id = list(np.unique(df_ind_s['study_id']))
        df_indc_s = pd.read_csv(self.in_indc_pth).sort_values(by='study_id').reset_index(drop=True)

        # prepare empty dataframes and dicts
        df_gen_findings_s = pd.DataFrame(columns=['study_id', 'report'])
        df_gen_imp_s = pd.DataFrame(columns=['study_id', 'report'])
        dict_ind_error = {}

        for i in tqdm(range(len(ls_id))): ###
            id = ls_id[i]
            n, repo_gen, dict_ind_error = self.gen_per_study(id, df_indc_s, df_ind_s, dict_ind_error)

            if n == self.max_retries-1 and "IMPRESSION" not in repo_gen: ## to distinguish formate error and api error
                if id not in dict_ind_error:
                    print(f's{id} still encounters request rejection after {n+1} tries: APIError')
                    dict_ind_error[id] = 'APIError'
                    continue    

            df_gen_findings_s, df_gen_imp_s, dict_ind_error = self.repo_split(repo_gen, id, dict_ind_error, df_gen_findings_s, df_gen_imp_s)

        # Save files of labels, repos, and errors
        df_gen_findings_s['study_id'] = df_gen_findings_s['study_id'].apply(str)
        df_gen_imp_s['study_id'] = df_gen_imp_s['study_id'].apply(str)
        df_gen_findings_s.to_csv(self.out_findings_pth, index=False)
        df_gen_imp_s.to_csv(self.out_imp_pth, index=False)

        temp_tuples = list(dict_ind_error.items())
        df_ind_err = pd.DataFrame(temp_tuples, columns=['study_id', 'error'])
        df_ind_err.to_csv(self.out_dir+f'err_list_{self.prompt_label}.csv', index=False)
    

if __name__ == '__main__':
  args, _ = parse_args()

  GenRepo = GenRepoAzureGPT(api_key=args.api_key,
                            endpoint=args.endpoint,
                            image_dir=args.image_dir, 
                            input_dir=args.input_dir, 
                            output_dir=args.output_dir, 
                            prompt_dir=args.prompt_dir, 
                            prompt=args.type, 
                            max_retries=3)
  GenRepo.run()