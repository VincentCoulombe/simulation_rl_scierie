paramSimu["NbStepSimulationTest"] = 64*10
paramSimu["NbStepSimulation"] = 64*10
lstUtilSechoir= []
lstRewards = []
for i in range(20) : #20 intervalles basés sur 64*10
print(i)
#lstRewards.append(env.solve_w_heuristique("aleatoire"))
env.evaluate_model(model)
lstRewards.append(env.rewards_moyens[-1])
lstUtilSechoir.append(env.env.getTauxUtilisationSechoirs())
print(lstRewards[-1])

print(lstUtilSechoir)
print(lstRewards)
print("Intervalles de confiances séchoirs : ",IntervalleConfiance(lstUtilSechoir))
print("Intervalles de confiances rewards: ",IntervalleConfiance(lstRewards))