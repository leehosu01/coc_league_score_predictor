#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:53:40 2020

@author: assign
"""
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *  
K = tf.keras.backend
def clip(X,value = .9):
    R = X*2 - 1
    R = -tf.where(R<value,R,tf.nn.tanh(R+tf.cast(tf.math.atanh(value),R.dtype)-value))
    R = -tf.where(R<value,R,tf.nn.tanh(R+tf.cast(tf.math.atanh(value),R.dtype)-value))
    return (R+1)/2
def score_make_rule(ATT,DEF):
    return (ATT / DEF) #**.5
def result_function(mean, std):
    def to_res(attack,defense):
        #res =  tf.expand_dims(attack,-1) /tf.expand_dims(defense,0)
        res =  score_make_rule(tf.expand_dims(attack,-1), tf.expand_dims(defense,0))
        return clip(res)
        return tf.nn.tanh(res)
    def to_res2(attack,defense):
        res = to_res(attack,defense)
        for _ in range(1):
            res = (res - tf.reduce_mean(res)) / tf.math.reduce_std(res) * std + mean
            res = clip(res)
        return res
    return to_res,to_res2
#@tf.function
def train_step( attack,defense ,target,mapped,opt,score_function,loss_function):
    with tf.GradientTape() as gp:
        res = score_function(attack,defense)
        losses = loss_function(target, tf.where(mapped>.5,tf.cast(res,target.dtype),target))
        R_loss = MAE(target, tf.where(mapped>.5,tf.cast(res,target.dtype),target))
    trainable_variables = [attack,defense]
    grad = gp.gradient(losses, trainable_variables)
    opt.apply_gradients(zip(grad, trainable_variables))
    return tf.reduce_sum(R_loss) / tf.cast(tf.reduce_sum(mapped),R_loss.dtype)
def training(attack,defense,to_res,epochs  = 1000):
    global target,mapped
    # training with noise 
    #opt = SGD(.01, momentum = .2)
    IRS = schedules.ExponentialDecay(.004,epochs//3,.25,True)
    opt = RMSprop(IRS)
    IRS = schedules.ExponentialDecay(.004,epochs//3,.25,True)
    opt = Adam(IRS,amsgrad = True)
    to_sco = to_res[0]
    loss_function = Huber(.03)#MSE#BinaryCrossentropy()#
    for epo in range(epochs):
        if epo == epochs - 100:
            #loss가 줄지는 않아도, 받아들일만한 값을 얻을수있다. 
            opt = RMSprop(.0001, momentum = .2,centered=False)
        noisy_target = clip(target+tf.random.normal(tf.shape(target),0,.03,dtype=target.dtype),.999)
        target_value =  noisy_target if epo != epochs-1 else target
        loss = train_step(attack,defense,target_value,mapped,opt,to_sco,loss_function)
    return attack,defense,loss
def output_state(loss = None , attack = None ,defense = None, result = None ):
    if loss is not None :print("loss = {}".format(loss))
    if  attack  is not None:print("attack = {}".format(tf.cast(attack*100,tf.int32)))
    if  defense  is not None:print("defense = {}".format(tf.cast(defense*100,tf.int32)))
    if  attack  is not None and defense  is not None:
        print("result = \n{}".format(tf.cast(to_res_tuple[0](attack,defense)*45,tf.int32)))
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
        i = teams[I]
        j = teams[J]
        
        if S1 != '':
            score = int(S1) / S_max
            result[i][j] = score # i공격력 , j방어력 => 획득율
            all_values.append(score)
            mapped[i][j] = 1
        if S2 != '':
            score = int(S2) / S_max
            result[j][i] = score # j공격력 , i방어력 => 획득율
            all_values.append(score)
            mapped[j][i] = 1
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
        wins = 0
        for i in range(len(teams)):
            wins += 1 if approximated[t][i] > approximated[i][t] else 0
            trops += approximated[t][i]
        return trops + 10 * wins , wins
    res = sorted([(*T_sco(i),I) for i,I in enumerate(teams)])
    return res[::-1]
#input of result T1 vs T2, S1 = star for T1, S2 = star for T2
# -> T1, T2, S1, S2
tuples = [('GG', 'MO', '30', '22')
          ,('TO', 'WA', '35', '24')
          ,('TR', 'CL', '32', '21')
          ,('IR', 'CH', '24', '22')
          
          #,('GG', 'TR', '28', '33')
          ,('WA', 'CH', '22', '29')
          ,('CL', 'TO', '14', '27')
          ,('IR', 'MO', '22', '30')
          
          
          ,('GG', 'CH', '31', '16')
          ,('CL', 'WA', '34', '26')
          ,('TR', 'MO', '28', '29')
          ,('IR', 'TO', '24', '37')
          
          #,('GG', 'TO', '31', '26')
          #,('CL', 'IR', '23', '28')
          #,('MO', 'CH', '28', '15')
          ,('TR', 'WA', '35', '19')
          ]
#global var
sampling = 10
teams = set()
all_values = []

# target = sco
# mapped = true if target is valid else False
target, mapped = convert_from_tuple(tuples,45)
to_res, _ = to_res_tuple = result_function(mean = tf.reduce_mean(all_values), std = tf.math.reduce_std(all_values) )
best_scenario = (1e9,1)

for _ in range(sampling):
    attack_constraint = lambda X: tf.clip_by_value(X, 0, 1)
    defense_constraint = lambda X: tf.clip_by_value(X,.001,.999)
    attack = tf.Variable(tf.random.uniform(shape = (8,), minval = 0.3, maxval = 0.6),trainable = True,constraint = attack_constraint)
    
    defense = tf.Variable(tf.random.uniform(shape = (8,), minval = 0.2, maxval = 0.7),trainable = True,constraint = defense_constraint)
    
    attack,defense,loss = training(attack,defense,to_res_tuple)
    output_state(loss,attack,defense)
    print("\n\n")
    if best_scenario[0] > loss:
        best_scenario = (loss,attack,defense)

# inputs
print("input stats",end = ' ')
output_state(result = tf.convert_to_tensor(target))
# predict
print("output stats",end = ' ')
loss,attack,defense = best_scenario
output_state(loss,attack,defense)
res = to_res(attack,defense)
print("input:: ",tf.reduce_mean(all_values),tf.math.reduce_std(all_values))
print("outss:: ",tf.reduce_mean(res),tf.math.reduce_std(res))
'''
#predict move_avr, rescale
Cres = to_res_tuple[1](attack,defense)
output_state(result = Cres)
'''
# league predict total trophy
print([(I[0].numpy(),I[1],I[2]) for I in each_team_trophies(res)] )
