

import pandas as pd

# remove columns that are not needed

def clean(data):
    to_drop = ['What US state or territory do you live in?',
           'What US state or territory do you work in?',
            'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
            'Why or why not?',
            'Did you feel that your previous employers took mental health as seriously as physical health?',
            'Do you think that discussing a physical health issue with previous employers would have negative consequences?',
            'Do you think that discussing a physical health issue with your employer would have negative consequences?',
           'Is your employer primarily a tech company/organization?',
           'Is your primary role within your company related to tech/IT?',
           'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
           'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:',
           'Would you feel comfortable discussing a mental health disorder with your coworkers?',
           'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
           'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?',
           'If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?',
           'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?',
           'If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?',
           'Have your previous employers provided mental health benefits?',
           'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?',
           'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?',
           'Do you think that discussing a mental health disorder with previous employers would have negative consequences?',
           'Do you think that discussing a physical health issue with previous employers would have negative consequences?',
           'Would you have been willing to discuss a mental health issue with your previous co-workers?',
           'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?',
           'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?',
           'Would you bring up a mental health issue with a potential employer in an interview?',
           'Why or why not?',
           'Do you feel that being identified as a person with a mental health issue would hurt your career?',
           'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?',
           'How willing would you be to share with friends and family that you have a mental illness?',
           'If yes, what condition(s) have you been diagnosed with?',
           'If maybe, what condition(s) do you believe you have?',
           'Have you ever sought treatment for a mental health issue from a mental health professional?',
           'What country do you live in?',
           'What US state or territory do you live in?',
           'What US state or territory do you work in?',
           'Do you work remotely?']

    data.drop(to_drop, inplace=True, axis=1)

    # gender column
    spellForFemale = ['f', 'F', 'woman', 'female', 'femail', 'Cisgender Female', 'fem', 'cis female',
                      'Female (props for making this a freeform field, though)',
                      'Female assigned at birth ', 'Female or Multi-Gender Femme', 'female/woman', 'fm', 'Cis-woman',
                      'I identify as female.', 'female ', 'Woman',
                      'Cis female', 'Genderfluid (born female)', 'Female', 'Female ', 'Cis female ', ' Female']

    spellForMale = ['Cis male', 'Cis Male', 'cisdude', 'Dude', 'man', 'M', 'm', 'M|', 'mail', 'Male (cis)', 'MALE',
                    'Male.', 'Malr', 'Sex is male', 'male',
                    'Male ', 'nb masculine', 'Man', 'cis male', 'male',
                    "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
                    'male ', 'cis man']

    spellOthers = ['AFAB', 'Agender', 'Androgynous', 'Bigender', 'Enby', 'female-bodied; no feelings about gender',
                   'Fluid', 'GenderFluid', 'Genderfluid (born female)'
                                           'Genderflux demi-girl', 'genderqueer', 'Genderqueer', 'Male (trans, FtM)',
                   'male 9:1 female, roughly', 'Male/genderqueer', 'mtf'
                                                                   'nb masculine', 'Nonbinary', 'non-binary', 'Other',
                   'Other/Transfeminine', 'Queer', 'Transgender woman', 'Transitioned, M2F', 'Unicorn',
                   'none of your business', 'Human', 'Genderfluid', 'genderqueer woman', 'mtf', 'Genderflux demi-girl',
                   'human']

    data.loc[0:, 'What is your gender?'].replace(spellForFemale, 'Female', inplace=True)
    data.loc[0:, 'What is your gender?'].replace(spellForMale, 'Male', inplace=True)
    data.loc[0:, 'What is your gender?'].replace(spellOthers, 'Other', inplace=True)




