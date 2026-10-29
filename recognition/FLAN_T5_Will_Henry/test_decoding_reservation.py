import unittest
from prepare_decoding_study import components, reserve


class ReservationTests(unittest.TestCase):
    def test_transitive_seen_component_excluded_and_samples_disjoint(self):
        rows=[{'id':str(i),'input':f'report {i}','image':f'image{i}'} for i in range(35)]
        rows[1]['input']=rows[0]['input'];rows[2]['image']=rows[1]['image']
        samples,receipt=reserve(rows,[rows[0]],representative=3,confirmation=3,
                                cutoff_representative=3,challenge=4,cutoff_challenge=3)
        self.assertEqual(receipt['excluded_groups'],1)
        flat=[r for values in samples.values() for r in values]
        self.assertEqual(len(flat),len({r['id'] for r in flat}))
        self.assertFalse({'0','1','2'} & {r['id'] for r in flat})
        # Every picked row belongs to a different full development component.
        selected={r['id'] for r in flat}
        self.assertTrue(all(sum(r['id'] in selected for r in g)<=1 for g in components(rows)))
        again,_=reserve(rows,[rows[0]],representative=3,confirmation=3,
                        cutoff_representative=3,challenge=4,cutoff_challenge=3)
        self.assertEqual(samples,again)


if __name__=='__main__':unittest.main()
