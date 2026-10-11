from experiments.semantic_assembly_20261009 import features,boundary
from llm_palindrome.admission import normalize_letters as norm

def row(sentences):return dict(sentences=sentences,repeated_sentences=0)
def sentence(frame,words,roles):return dict(frame=frame,words=words,roles=roles)
def test_shared_person_does_not_create_progression():
 r=row([sentence('past_revile',['Anna','reviled','Noel'],['human_agent','past_verb','human_theme']),sentence('past_revile',['Noel','reviled','Anna'],['human_agent','past_verb','human_theme'])])
 assert features(r)['link_count']==0

def test_directed_recipient_to_affected_actor():
 a=sentence('food_delivery_request',['Noel','deliver','Anna','desserts'],['addressee','imperative','recipient','food_theme'])
 b=sentence('stressed_criticism',['stressed','Anna','reviled','Noel'],['human_adjective','human_agent','past_verb','human_theme'])
 assert features(row([a,b]))['link_count']==1
 b['words'][1]='Eve';assert features(row([a,b]))['link_count']==0

def test_center_insertion_preserves_exactness():
 ss=[{'text':'Noel, deliver Anna desserts.'},{'text':'Stressed Anna reviled Leon.'}]
 r={'sentences':ss,'letters':len(norm(' '.join(s['text'] for s in ss)))}
 assert boundary(r)==1
 inner='Step on no pets.';text=ss[0]['text']+' '+inner+' '+ss[1]['text']
 assert norm(text)==norm(text)[::-1]
