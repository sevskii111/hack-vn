import re
import json
import pandas as pd
from tqdm import tqdm
from fastai.text.all import *
import gdown

urls = 'https://drive.google.com/file/d/1-PvRdDhEM9KQOmoS_oQQWsuFWn-VvkSb/view?usp=sharing, https://drive.google.com/file/d/10dABoxzI0Z62L9cnqYeJ_HbANTz5_9rz/view?usp=sharing, https://drive.google.com/file/d/11VYYTodHGgA9vTT5D5aR5AGeythZLcXh/view?usp=sharing, https://drive.google.com/file/d/12Qhx6koxh2xEbRFWYzThoWrHWuYQBv7B/view?usp=sharing, https://drive.google.com/file/d/13QZtCoQu4B0yWiIdFICvZoywYHoB9CJa/view?usp=sharing, https://drive.google.com/file/d/14jeVdyahRpePB414WcL3iPFeQmtOjhUn/view?usp=sharing, https://drive.google.com/file/d/1572xb-HvOjyo3wAskC_txeWgHMULVYVK/view?usp=sharing, https://drive.google.com/file/d/15uSYUylgiVNmTGK2AMmKqpNA_7N2kfud/view?usp=sharing, https://drive.google.com/file/d/18JZHUarxLBJhmOFmYhx4kWr58lkk2cAV/view?usp=sharing, https://drive.google.com/file/d/18z6N4WjwJMIvPW6Or02v9SpdX_folu8V/view?usp=sharing, https://drive.google.com/file/d/18zqIyUTjeQ1wJ9uwe22uVvMeax3SDKwi/view?usp=sharing, https://drive.google.com/file/d/19QfeqH8QNVYJKjOn9q3aTqA96yshWUqO/view?usp=sharing, https://drive.google.com/file/d/19enPL6Td_fT80Bwe37iuY7tjpytmMOl-/view?usp=sharing, https://drive.google.com/file/d/1A_X_BDhJ7w2J3qocF80j9D4E0hcZ8kv-/view?usp=sharing, https://drive.google.com/file/d/1COuBAnbMIj-GRwwvv5VZYKgwQ7KHehrA/view?usp=sharing, https://drive.google.com/file/d/1CUSk8dxs_JgQOqLYINUf21akuSqMOwqa/view?usp=sharing, https://drive.google.com/file/d/1DA78Ampb4X_fPzpoK8b-gDVKlnVz3e91/view?usp=sharing, https://drive.google.com/file/d/1DQ7Z5MDIQfc07PtRSFxcklDDBRXjMRVB/view?usp=sharing, https://drive.google.com/file/d/1E0oVZkhNjgzAwwjsYZ5yC_s_yD6f9q0c/view?usp=sharing, https://drive.google.com/file/d/1EJ4ROandl-bhs_tebAKHciE4Oy5OooDh/view?usp=sharing, https://drive.google.com/file/d/1EK1I9eUEq3VTWf0JhiDlMZ5ZVbtuH0t2/view?usp=sharing, https://drive.google.com/file/d/1HDwvn3_FHA6cO7eXctYn2LENThndIalv/view?usp=sharing, https://drive.google.com/file/d/1HJA4fknJJNrUM03mSujSGsf3SyWLdHF3/view?usp=sharing, https://drive.google.com/file/d/1HckcVEezFFlD-0ByQ5eR2XRMm2fuGAKU/view?usp=sharing, https://drive.google.com/file/d/1IphgWz1-Wj3uozm6UjoDvXa90jk4iuu6/view?usp=sharing, https://drive.google.com/file/d/1JVfNjcQ83s-CB964xcUIGu--otH2Ycaj/view?usp=sharing, https://drive.google.com/file/d/1JkkilEVRqAVa5WCEZbx0IKHqijRFtSOM/view?usp=sharing, https://drive.google.com/file/d/1K7JZS4ngdbqh192AKp_wlzDMrNkBt7Ld/view?usp=sharing, https://drive.google.com/file/d/1LJ5RRrerKSDfc-T2SfeWRJFjyT9GMukp/view?usp=sharing, https://drive.google.com/file/d/1LUwgDOmzN4yGUAI1k0HKOE9_uD_hU7I_/view?usp=sharing, https://drive.google.com/file/d/1Lfymo45KapKjbN66_d6LiPNOmGYvFFkq/view?usp=sharing, https://drive.google.com/file/d/1M6caCM2lbLnBmQV4KAS_lCSt48sROSBL/view?usp=sharing, https://drive.google.com/file/d/1NE6pxIUikF3YnletZs2cFgrhS7iNEnck/view?usp=sharing, https://drive.google.com/file/d/1OvwlgRpZVmwIoOnE47zsn8ziPHkuHfOv/view?usp=sharing, https://drive.google.com/file/d/1PMPhM5Jb8GI6k71V9yhraokvtsGd5-gH/view?usp=sharing, https://drive.google.com/file/d/1PNp-VooEuJSi1WnYWzUM-pFms8WbhOjn/view?usp=sharing, https://drive.google.com/file/d/1QBWmun6McQuz16RoSIsbZxbs-VCPPSHJ/view?usp=sharing, https://drive.google.com/file/d/1SktRd4uvCXchl5BcwFthJIymMgtBR1KO/view?usp=sharing, https://drive.google.com/file/d/1SuFpYt4WeQ8O8gcZhnaDmwejBoTc0FDW/view?usp=sharing, https://drive.google.com/file/d/1TUsOMdaqULCzS868SkLU8QjW7ZzaW3a1/view?usp=sharing, https://drive.google.com/file/d/1TlFymvftjvL2gD7cp7ziZh453OpxXk_K/view?usp=sharing, https://drive.google.com/file/d/1VD9R_55xQWfDkT1b-H7CuIEj2gDJa2ij/view?usp=sharing, https://drive.google.com/file/d/1X9vw8lVAuDHYVkF1rxatkDKo7KvAEU_5/view?usp=sharing, https://drive.google.com/file/d/1XOwquh-DyX4inVjgjhj_gljjaU9CFcqM/view?usp=sharing, https://drive.google.com/file/d/1ZCpyRQhk3ac8uHxaAvirnQBEXOycpwES/view?usp=sharing, https://drive.google.com/file/d/1_5VzmtF8fF816GCOLeThPI1kwJ3WH4O3/view?usp=sharing, https://drive.google.com/file/d/1_Zpjogjgq_VVd82SqHZCwuyE2_OphV5m/view?usp=sharing, https://drive.google.com/file/d/1a8VRliWTJOOfWkTx1_50PmJ04PF2rwnR/view?usp=sharing, https://drive.google.com/file/d/1cL6SRThH6n4AP-VpOGNuxGRmBO9lly4f/view?usp=sharing, https://drive.google.com/file/d/1ckmfFIBZqkRsA3zJgobn5mFf9sLFBiJd/view?usp=sharing, https://drive.google.com/file/d/1gRCKFYqv7G0izZyTAaHI4pzemVBBwShz/view?usp=sharing, https://drive.google.com/file/d/1iW1P4yV1EFpYFYNHRfMUzl0S1QcSIK7E/view?usp=sharing, https://drive.google.com/file/d/1jK2C7DYJmgYyGcbYP2aNY4iAjdtPKeIT/view?usp=sharing, https://drive.google.com/file/d/1jOX0bo6iOTJOFtvIIT2F8vwpvhDWJXx_/view?usp=sharing, https://drive.google.com/file/d/1jpS93gAIRvZh-mo9x4LsG_bvGuiKASju/view?usp=sharing, https://drive.google.com/file/d/1k8lgl4odrQ3kAFlKKewhhX5nTdVDINFm/view?usp=sharing, https://drive.google.com/file/d/1kYjp6F8LEcQ28Z0Hc-Vn8iDv_4Rc4_iX/view?usp=sharing, https://drive.google.com/file/d/1loh9KERC5f05LLxV5wG54040OVfAtdWt/view?usp=sharing, https://drive.google.com/file/d/1ndUFU-AR3o2exxjKOAdhL0VLx-2Fm9Wi/view?usp=sharing, https://drive.google.com/file/d/1nlec4v-_7srRgo1t8kdcK_n9m948XhUC/view?usp=sharing, https://drive.google.com/file/d/1pKxfd9R6T38dfTWDvPwNcTQMZhoGC65i/view?usp=sharing, https://drive.google.com/file/d/1qHm5CawTQVT2OQLGJXxNtVpYBYWl6z48/view?usp=sharing, https://drive.google.com/file/d/1qus3O1kai3e-lrkQM4AethyIjPhDlUQ8/view?usp=sharing, https://drive.google.com/file/d/1qwitXvfg26mP9v5uJ6TErXsGaph_ev4X/view?usp=sharing, https://drive.google.com/file/d/1rbh4OG62Dwy4MEXzJetJ3Hpkc_ktzl-e/view?usp=sharing, https://drive.google.com/file/d/1rwTPYtwpgnqIwqJJK2cdxkZj2rCCNJ4L/view?usp=sharing, https://drive.google.com/file/d/1rzRYHY3q06xnx4YNQLBPJ94n2I2N1ARc/view?usp=sharing, https://drive.google.com/file/d/1sHMlZFdYM27BoDrcVqhvB1EzvtBGIWOG/view?usp=sharing, https://drive.google.com/file/d/1sJZ4mPBnJ0rax4TufyxKngi0-Cvbc1Ws/view?usp=sharing, https://drive.google.com/file/d/1sWpw_ke0taCdJSo4D7F7vWaeDtmCYnR9/view?usp=sharing, https://drive.google.com/file/d/1tD9QuXUbAJdHqIHDPgS8owiR2UWLEkAl/view?usp=sharing, https://drive.google.com/file/d/1tmbbXWdZTAvERAyIRfFY-lEBVjYdXncX/view?usp=sharing, https://drive.google.com/file/d/1ujIiL6MP4LWnZT8KG0qW7-9zvGs-ylFj/view?usp=sharing, https://drive.google.com/file/d/1urLPXjiRoP_fRgVC-D7z4ihl32GB7ueQ/view?usp=sharing, https://drive.google.com/file/d/1v-oGcCWtCl1Mcs6u5x30aRg23XQO9IzD/view?usp=sharing, https://drive.google.com/file/d/1v0sHcxbpt37GxXt4oZC1u2OwFE9vtPce/view?usp=sharing, https://drive.google.com/file/d/1vd16lV7es63lacOA9dnPf5GfV1_TjR2-/view?usp=sharing, https://drive.google.com/file/d/1w95vGaYUhmfntmPOD0uch0OeUP2c3rBp/view?usp=sharing, https://drive.google.com/file/d/1wIOxLW2kISM36jJ4LsPMRz-LCDW4-JcF/view?usp=sharing, https://drive.google.com/file/d/1yKNwY06iZxHbrWf87vsNRhlw2KPON9S-/view?usp=sharing, https://drive.google.com/file/d/1yZx7mDspnT0hmofgDhSYy0xa1kBVzoxZ/view?usp=sharing'.split(',')

for url in urls:
    file_id = re.search(r"d/([^/]+)", url).group(1)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, quiet=False)
    print(file_id)


samples = []
for i in tqdm(range(79 + 1)):
    file_path = Path(f"./documents_{i}.json")
    if file_path.is_file():
        with open(file_path) as las_json:
            las = json.load(las_json)
            for la in las:
                samples += [line for line in la["additionalFields"][1]["value"].split('\n\n') if len(line) > 100]


texts_df = pd.DataFrame(samples, columns=["text"]).sample(n=4000000, replace=False, random_state=42)
tok = WordTokenizer('ru')
dblocks = DataBlock(blocks=(TextBlock.from_df('text', tok=tok, is_lm=True)),
                    get_x=ColReader('text'), 
                    splitter=RandomSplitter(0.05))
dls = dblocks.dataloaders(texts_df, bs=64)
learn = language_model_learner(
    dls, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()], pretrained=False).to_fp16()
learn.fit_one_cycle(1, 2e-2)
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)

learn.save_encoder('finetuned')
with open('./models/vocab.pkl', 'wb') as file:
    file.write(dls.vocab)