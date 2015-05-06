
"""
ksig_dict is a dictionary that looks up key signature by the number of flats and sharps: ksig_dict[number_of_sharps][number_of_flats]
"""
ksig_dict={}
ksig_dict['sharps']={}

ksig_dict['sharps'][0]='c'
ksig_dict['sharps'][1]='g'
ksig_dict['sharps'][2]='d'
ksig_dict['sharps'][3]='a'
ksig_dict['sharps'][4]='e'
ksig_dict['sharps'][5]='b'
ksig_dict['sharps'][6]='f#'
ksig_dict['sharps'][7]='c#'
ksig_dict['sharps'][8]='g#'

ksig_dict['flats']={}

ksig_dict['flats'][0]='c'
ksig_dict['flats'][1]='f'
ksig_dict['flats'][2]='b-'# - denotes flat, so this is bflat
ksig_dict['flats'][3]='e-'
ksig_dict['flats'][4]='a-'
ksig_dict['flats'][5]='d-'
ksig_dict['flats'][6]='f#'
ksig_dict['flats'][7]='c-'
ksig_dict['flats'][8]='f-'

half_steps_from_c={}
half_steps_from_c['c']=0
half_steps_from_c['g']=
half_steps_from_c['d']=
half_steps_from_c['a']=
half_steps_from_c['e']=
half_steps_from_c['b']=
half_steps_from_c['f#']=
half_steps_from_c['c#']=
half_steps_from_c['g#']=
half_steps_from_c['f']=
half_steps_from_c['b-']=
half_steps_from_c['e-']=
half_steps_from_c['a-']=
half_steps_from_c['d-']=
half_steps_from_c['c-']=
half_steps_from_c['f-']=