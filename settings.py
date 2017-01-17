# settings.py


# Algorithm settings
settingsDict = {

'granularity' :         1,
'training_window' :    24,
'training_interval' :   1,
'ema_alpha' :         1.0,
'severity_lambda' :   1.0,
'severity_omega' :    4.0,

}

# List any features here that should NOT be used
# This will be overwritten if the whitelist is non-empty
blacklist = [

]

# List features that should ONLY be used
# This list takes priority over the blacklist
whitelist = [

]

