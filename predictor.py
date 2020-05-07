#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:53:40 2020

@author: assign
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *  
def result_function(mean, std):
    def to_res(attack,defense):
        #return tf.minimum(1.0,attack/defense)
        res =  tf.expand_dims(attack,-1)/tf.expand_dims(defense,0)
        return tf.nn.tanh(res)
        return tf.clip_by_value(res,0,1)
        return tf.minimum(1.0,res)
    return to_res
#@tf.function
def train_step( attack,defense ,target,mapped,opt,score_function):
    with tf.GradientTape() as gp:
        res = score_function(attack,defense)
        diff = BinaryCrossentropy()(target, res)
        losses = tf.where(mapped>.5,diff,tf.zeros_like(diff))
    trainable_variables = [attack,defense]
    grad = gp.gradient(losses, trainable_variables)
    opt.apply_gradients(zip(grad, trainable_variables))
    return tf.reduce_mean(losses)
def training(attack,defense,epochs  =500):
    global target,mapped
    # training with noise 
    opt = Nadam(0.0003)
    for epo in range(epochs):
        attack = tf.Variable(tf.clip_by_value(attack,0,1),trainable = True,name = 'att')
        defense = tf.Variable(tf.clip_by_value(defense,1e-6,1),trainable = True,name = 'def')
        
        noisy_target = tf.clip_by_value(target+tf.random.normal(tf.shape(target),0,.1,dtype=target.dtype),0,1)
        target_value =  noisy_target if epo != epochs-1 else target
        loss = train_step(attack,defense,target_value,mapped,opt,to_res)
        
        attack = tf.Variable(tf.clip_by_value(attack,0,1),trainable = True,name = 'att')
        defense = tf.Variable(tf.clip_by_value(defense,1e-6,1),trainable = True,name = 'def')
    return attack,defense,loss
def output_state(loss = None , attack = None ,defense = None, result = None ):
    if loss is not None :print("loss = {}".format(loss))
    if  attack  is not None:print("attack = {}".format(tf.cast(attack*100,tf.int32)))
    if  defense  is not None:print("defense = {}".format(tf.cast(defense*100,tf.int32)))
    if  attack  is not None and defense  is not None:
        print("result = \n{}".format(tf.cast(to_res(attack,defense)*45,tf.int32)))
    elif result is not None:
        print("result = \n{}".format(tf.cast(result*45,tf.int32)))
def convert_from_tuple(tuples,S_max):
    global teams,all_values
    for I,J,S1,S2 in tuples:
        teams.add(I)
        teams.add(J)
    teams = {I:i for i,I in enumerate(sorted(teams) )}
    
    result = np.zeros((len(teams),len(teams)) )
    mapped = np.zeros((len(teams),len(teams)) )
    print("teams : {}".format(teams))
    for I,J,S1,S2 in tuples:
        s1 = int(S1)
        s2 = int(S2)
        i = teams[I]
        j = teams[J]
        result[i][j] = s1 / S_max# i공격력 / j방어력 
        result[j][i] = s2 / S_max# j공격력 / i방어력 
        all_values.append(s1 / S_max)
        all_values.append(s2 / S_max)
        mapped[i][j] = mapped[j][i] = 1
    print(result)
    return tf.convert_to_tensor(result),tf.convert_to_tensor(mapped)
def verses_score(res,T1,T2):
    def defence_std(tt):
        std = [tf.cast(target[i][tt],tf.float32) - tf.cast(res[i][tt],tf.float32)\
            for i in range(len(teams)) if mapped[i][tt] > .5]
        return np.std(std)
        
    global teams,target, mapped
    t1,t2 = teams[T1], teams[T2]
    print("T1 score = {}, std = {}".format(res[t1][t2]*45, defence_std(t2)*45))
    print("T2 score = {}, std = {}".format(res[t2][t1]*45, defence_std(t1)*45))
def each_team_trophies(res):
    global teams,target, mapped
    approximated = tf.where(mapped>.5,target,tf.cast(res,target.dtype)) * 45
    def T_sco(t):
        trops = 0
        for i in range(len(teams)):
            trops += 10 if approximated[t][i] > approximated[i][t] else 0
            trops += approximated[t][i]
        return trops
    res = sorted([(T_sco(i),I) for i,I in enumerate(teams)])
    return res[::-1]
#input of result T1 vs T2, S1 = star for T1, S2 = star for T2
# -> T1, T2, S1, S2
tuples = [('GG', 'MO', '30', '22'),
          ('TO', 'WA', '35', '24'),
          ('TR', 'CL', '32', '21'),
          ('IR', 'CH', '24', '22'),
          
          ('GG', 'TR', '28', '33'),
          ('WA', 'CH', '22', '29'),
          ('CL', 'TO', '14', '27'),
          ('IR', 'MO', '22', '30'),
          
          ('GG', 'CH', '31', '16'),
          ('CL', 'WA', '34', '26'),
          ('TR', 'MO', '28', '29'),
          ('IR', 'TO', '24', '37'),
          
          ('TR', 'WA', '35', '10')]
#global var
sampling = 10
teams = set()
all_values = []

# target = sco
# mapped = true if target is valid else False
target, mapped = convert_from_tuple(tuples,45)
to_res = result_function(mean = tf.reduce_mean(all_values), std = tf.math.reduce_std(all_values) )
best_scenario = (1e9,1)

for _ in range(sampling):
    attack = tf.Variable(tf.random.uniform(shape = (8,)),trainable = True)
    
    defense = tf.Variable(tf.random.uniform(shape = (8,)),trainable = True)
    
    attack,defense,loss = training(attack,defense)
    output_state(loss,attack,defense)
    print("\n\n")
    if best_scenario[0] > loss:
        best_scenario = (loss,attack,defense)
loss,attack,defense = best_scenario
output_state(loss,attack,defense)

loss,attack,defense = best_scenario
res = to_res(attack,defense)

print("input:: ",tf.reduce_mean(all_values),tf.math.reduce_std(all_values))
print("outss:: ",tf.reduce_mean(res),tf.math.reduce_std(res))
Cres = res
for _ in range(10):
    Cres = (Cres - tf.reduce_mean(Cres)) / tf.math.reduce_std(Cres) * tf.math.reduce_std(all_values) + tf.reduce_mean(all_values)
    Cres = tf.clip_by_value(Cres,0,1)
# inputs
output_state(result = tf.convert_to_tensor(target))
# predict
output_state(result = Cres)
# league predict total trophy
print(each_team_trophies(Cres) )
