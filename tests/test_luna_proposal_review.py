import unittest
from llm_palindrome.block_seams import Seam,Piece
from llm_palindrome.luna_proposal_review import decode_response,review_response


class NativeProposalTests(unittest.TestCase):
    def setUp(self):
        self.req=dict(request_id='synthetic-test',left_surface='',right_surface='in a cave',
                      seam_debt=Seam((),(Piece('right',0,'in a cave'),)).debt(),admissible_sides=['left'])
    def response(self,text):
        return dict(request_id='synthetic-test',proposals=[dict(side='left',decoded_text=text,
                   block_id='synthetic',score=0,provenance='synthetic test, no Luna call')])
    def test_feasible_decoded_text_is_not_grammar_certification(self):
        r=review_response(self.req,self.response('Eva can inspect'))
        row=r['reviewed_proposals'][0]
        self.assertEqual(row['status'],'letter_feasible_grammar_pending')
        self.assertFalse(row['grammar_verified']);self.assertFalse(r['model_invoked_by_checker'])
        self.assertEqual(row['new_debt']['residual'],'nspect')
    def test_retain_rejected_and_no_human_labels(self):
        r=review_response(self.req,self.response('Nora opens'))
        self.assertEqual(r['reviewed_proposals'][0]['status'],'rejected')
        self.assertIsNone(r['human_feedback'])
        r=review_response(self.req,self.response('Eva can inspect'),grammar_accept=lambda s,p:False)
        self.assertEqual(r['reviewed_proposals'][0]['reason'],'unlicensed grammar continuation')
    def test_schema_id_side_and_false_numeric_rank_rejected(self):
        bad=self.response('Eva can inspect');bad['request_id']='other'
        with self.assertRaises(ValueError):decode_response(bad,self.req)
        bad=self.response('Eva can inspect');bad['proposals'][0]['score']=True
        with self.assertRaises(ValueError):decode_response(bad,self.req)
        bad=self.response('Eva can inspect');bad['proposals'][0]['side']='right'
        with self.assertRaises(ValueError):decode_response(bad,self.req)


if __name__=='__main__':unittest.main()
