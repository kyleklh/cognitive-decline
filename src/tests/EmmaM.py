# library that allows set of fuctions for JSON (ie dumps)
import json
# congnitive decline segementation model
#States:
# - S1: Normal Behavior 
# - S2: Exit-seeking behavior
# - S3: Exit-Seeking 
# Observation vector:
# [distance_to_exit, dwell_time, entry_count, velocity_toward_exit]
#Output:
# JSON records with timestamp, state, confidence and features
# 1. State names ""
states = ["S1_Normal","S2_Borderline","S3_ExitSeeking"]
# 2. Starting Prob; tells model a state likely at begining 
start_prob = {"S1_Normal" : 0.8, # 80% normal 
              "S2_Borderline" : 0.15,  # 15% borderline 
              "S3_ExitSeeking" : 0.05}  # 5% exit seeking
# 3. Transition Probabilties; tell use how behavior changes over time
transition = {"S1_Normal" :{ #if someone display noraml behavior then begin with 
        "S1_Normal" : 0.70,
        "S2_Borderline" : 0.25,
        "S3_ExitSeeking" : 0.05
    },
    "S2_Borderline":{
        "S1_Normal" : 0.20,
        "S2_Borderline" : 0.50,
        "S3_ExitSeeking" : 0.30
    },
    "S3_ExitSeeking":{
        "S1_Normal" : 0.05,
        "S2_Borderline" : 0.20,
        "S3_ExitSeeking" : 0.75
    }
}
# Observations; your data; measurement taken from SAM
# stores as a dictionary: as each pair consist on a key(word) and the asscoiates number seperate by a :
#describe what system boserved in each timestamp/ frame create the input for HHM
observations = [
    {"timestamp": 0, "distance": 2.8, "dwell": 0.2, "numEntry": 0, "velocity": 0.1},
    {"timestamp": 1, "distance": 2.2, "dwell": 0.5, "numEntry": 0, "velocity": 0.2},
    {"timestamp": 2, "distance": 1.6, "dwell": 1.2, "numEntry": 1, "velocity": 0.4},
    {"timestamp": 3, "distance": 1.1, "dwell": 2.0, "numEntry": 1, "velocity": 0.6},
    {"timestamp": 4, "distance": 0.7, "dwell": 3.2, "numEntry": 2, "velocity": 0.8},
    {"timestamp": 5, "distance": 0.3, "dwell": 4.5, "numEntry": 3, "velocity": 1.0}
]
# Emission Probability 
#Calculates most likely state based on the measurements (values in observations)
def get_emission_prob(obs):
# Pull features out of a single dictionary 'obs'
# extracts values
    distance = obs["distance"]
    dwell = obs["dwell"]
    numEntry = obs["numEntry"]
    velocity = obs["velocity"]
#Normal Score
    normal = 0
    if distance > 2: normal += 1 # far from exit normal
    if dwell < 1: normal += 1 #not staying near door
    if numEntry == 0: normal += 1 #never entering exit zone
    if velocity < 0.3: normal += 1 #moving slowly
    #each true condition add 1 point to normal score
#Borderline Score
    borderline = 0
    if 0.8 < distance <= 2: borderline += 1 # closer to exit
    if 1 <= dwell <= 3: borderline += 1 # longer time near exit
    if 1 <= numEntry <= 2: borderline += 1 # more entries
    if 0.3 <= velocity <= 0.8: borderline += 1 # moving faster
    #each true add to borderline
#Exit Seeking
    exitSeek = 0
    if distance <= 0.8: exitSeek += 1 # very close to exit
    if dwell > 3: exitSeek += 1 # longer time near exit
    if numEntry >= 3: exitSeek += 1# more entries
    if velocity > 0.8: exitSeek += 1# moving faster
    # each true ass to exit seeking
# create probability dictionary (add 0.1 so prob is ever a hard 0)
# Create a dictonary for prob and asscociates values to each state 
    probability = {
        "S1_Normal": normal + 0.1,
        "S2_Borderline": borderline + 0.1,
        "S3_ExitSeeking": exitSeek + 0.1
}
#turn score in to percentage
# calculate percentages; need total to divide
    total = sum(probability.values())
    for s in probability:
        # calculates percentage for each key in dictionary 
        probability[s] = probability[s] / total
    # returns filled array 
    return probability

# Run HMM fucntion 
def run_HMM(observations):
    # starts with intial probability, creates a copy stored under prevProb so values
    #  can be modified without changing orignal source if start_prob
    prevProb = start_prob.copy()
    #create empty array for results to be stored
    results = []

    # loops for each timestamp in observations
    for obs in observations:
        # calculates the probabilites for each time stamp sending each key in observations to function
        emissionProb = get_emission_prob(obs)
        # creates empty dictonary to store currentProb
        currentProb = {}
    # loops through each state
        for current_state in states:
            # intialize best score at 0
            bestScore = 0
            # Check every possible previous states
            for previous_state in states:
                # Calculate probability of being in current state 
                # score = prevProb * transition Prob * observation prob 
                score = (prevProb[previous_state]* #Prob in this prev state
                         transition[previous_state][current_state] #prob of moving from prev state to current state
                        *emissionProb[current_state]) # probability that observations fits current state
                
                # if this is the best score found; then that is the leading probable state 
                if score > bestScore:
                    # updates score
                    bestScore = score
            # after checking alll prev states; stores the prob 
            currentProb[current_state] = bestScore
    
        #Normalize prob
        # Ensures all prob add up to 1 
        # total is the sum of all current prob
        total = sum(currentProb.values())
        # loops through each state in dictionary
        for s in currentProb:
        # Scale each probability so that the total = 100% / 1
            currentProb[s] = currentProb[s] / total
        prevProb = currentProb
        
    # Pick the most likely state
        # Looks throught the dictionary and pic the state with the highest prob score
        # max finds biggest item, input = currentProb (list of all states), rule = (key.currentProb.get; get the number)
        bestState = max(currentProb, key=currentProb.get)
        # get the prob score and round to 4 deci places
        confidence = round(currentProb[bestState], 4)
#takes empty list of result and adds to the list
# each observations creates and entry 
    results = {
        # store the time step from the observation
        "finalTimestamp" : obs["timestamp"],
        # store the predicted behavior state (Normal, Borderline, ExitSeeking)
        "state": bestState,
        # Score condifence of the model in that prediction
        "confidence": confidence,
        #saves the feautes used to make the descision
        "finalFeatures": {
            "distanceToExit": obs["distance"],
            "dwellTimeNearExit": obs["dwell"],
            "numberOfEntries" : obs["numEntry"],
            "velocityTowardExit": obs["velocity"]
        }
    }
    #return full list of predictions 
    return results

def main():
    # run the HMM func and return a list of results
    results = run_HMM(observations)
    print("HMM Results: \n")
    # loops thru each result in list
    print(
        #prints out values of each result
        f"t={results['finalTimestamp']} | "      # the time step
        f"state={results['state']} | "      # predicted hidden state
        f"confidence={results['confidence']}"  # probability of that state
        )
        #print out JSON version of result
    print("\n Json Output: \n")
    print(json.dumps(results, indent=4))
        #save results to a file calles "output JSON" and writes ("w" writiing mode)
    with open("output Json", "w") as file:
        json.dump(results, file, indent=4)
            #tells user that the file was saved 
    print("\n Sucessfully Saved\n")

#prvent code from running auto; oinly runs if script is excuted 
if __name__ == "__main__":
    main()


