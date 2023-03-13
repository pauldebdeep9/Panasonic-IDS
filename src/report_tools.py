def report_counts(f_infos):
    print('# Causal rels in data:', len(f_infos))
    counter = 0
    for k,v in f_infos.items():
        counter+=len(v)
    print('# Causal examples in data:', counter)
    print('Average support per rel:', counter/len(f_infos))


def report_intersection(f_infos):
    unicausal_count  = 0
    unicausalp_count = 0
    unicausalm_count = 0
    causenet_count = 0
    overlap_count = 0
    total = 0
    for k,v in f_infos.items():
        for vv in v:
            if len(vv['method'])==2:
                unicausal_count += 1
                causenet_count += 1
                overlap_count += 1
                total+=2
            else:
                if 'unicausal' in vv['method']:
                    unicausal_count += 1
                elif 'unicausal+' in vv['method']:
                    unicausalp_count += 1
                elif 'unicausalm' in vv['method']:
                    unicausalm_count += 1
                elif 'causenet' in vv['method']:
                    causenet_count += 1
                else:
                    raise ValueError('There cannot be a relation without any method support!')
                total+=1
    print(f'UniCausal: {unicausal_count}, UniCausal+: {unicausalp_count}, UniCausalM: {unicausalm_count}, CauseNet: {causenet_count}')
    print(f'Intersection: {overlap_count}, Union: {unicausal_count+causenet_count-overlap_count}')
    print(f'Total Evidence Counts: {total}')
