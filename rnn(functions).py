import warnings
from torch.autograd import Function, NestedIOFunction, Variable
from torch.autograd.function import _iter_variables, _unflatten
import torch.backends.cudnn as cudnn
from .. import functional as F
from .thnn import rnnFusedPointwise as fusedBackend
from sklearn.cluster import DBSCAN
import pandas as pd 
import torch 
torch.backends.cudnn.enabled = False

try:
    import torch.backends.cudnn.rnn
except ImportError:
    pass

def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy
#####################################################################################################################################
setting = 6

setting_6_version = 1

print ('setting: ', setting)
setting_2_version = 2
setting_3_version = 2
setting_4_control = False #RUNS THE CONTROL FOR SETTING 4: either True or False  (ADAPTIVE ACTIVATION FUNCTION)


'for setting 9:'
preset = False #either True or False (RELATED TO INCOMPLETE IDEA)
effect = 'network' #if effect is on cellgate ('cellular') or short term memory ('network')
#######################################################################################################################################
if setting == 1: #NORMAL STANDARD LSTM CELL
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
#        print('working')
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh) 
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)         
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
    
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
    
        return hy, cy
#######################################################################################################################################
if setting == 2: #TAKES WEIGHTED INPUT FROM 2 ACTIVATIONS
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        
        hx, cx = hidden

        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh) 
        
        if setting_2_version == 1: #LACK OF INFORMATION = NOISE     + DIST GATE                                
           
            ingate, forgetgate, cellgate, outgate, noisegate, distributiongate = gates.chunk(6, 1)  
            d_gate = F.relu(distributiongate)
            noise = .5*(torch.autograd.Variable(torch.randn(d_gate.size())))
            noisegate = F.relu(noisegate)*noise 
            noise = (noise/(d_gate+1e-10) + (.1/(noise+1e-10))*d_gate)/(d_gate + 1/(d_gate + 1e-10))
            noise = F.tanh(noise/(cellgate**2 + .2))                                    
            cellgate = F.tanh(cellgate+noisegate)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
         
            cy = (forgetgate * cx) + (ingate*cellgate)
            hy = outgate * F.tanh(cy)
       
        if setting_2_version == 2: #LACK OF INFO = NOISE (RAND**3) 
            ingate, forgetgate, cellgate, outgate, noisegate = gates.chunk(5, 1)  
            noise = torch.autograd.Variable((torch.rand(cellgate.size())-.5)**3)
            noisegate = F.relu(noisegate)*F.tanh(noise)
            cellgate = F.tanh(cellgate + noisegate)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)     

            cy = (forgetgate * cx) + (ingate*cellgate)
            hy = outgate * F.tanh(cy)
            
        if setting_2_version == 3: #ELABORATE NOISE    
            ingate, forgetgate, cellgate, outgate, noisegate, distributiongate = gates.chunk(6, 1)  
            d_gate = F.relu(distributiongate)
            noise = torch.autograd.Variable(torch.randn(d_gate.size()))
            noise = (noise/(d_gate+1e-10) + (1/(noise+1e-10))*d_gate)/(d_gate + 1/(d_gate + 1e-10))
            noisegate = F.relu(noisegate)*noise                         
            cellgate = F.tanh(cellgate+noisegate)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
         
            cy = (forgetgate * cx) + (ingate*cellgate)
            hy = outgate * F.tanh(cy)
        return hy, cy
#####################################################################################################################################
if setting == 3: #TAKES INPUT FROM 4 ACTIVATIONS OR ADAPTS SHORT TERM MEMORY TO 2 ACTIVATIONS
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)       
        hx, cx = hidden
    
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh) 
        
        if setting_3_version == 2: #4 ACTIVATIONS
            ingate, forgetgate, cellgate, outgate, choosegate_A, choosegate_B, choosegate_switch_1, stretchgate = gates.chunk(8, 1) 
             
            stretchgate = F.tanhshrink(stretchgate)
            choosegate_switch_1 = F.sigmoid(choosegate_switch_1).round()
            choosegate_A = F.sigmoid(choosegate_A).round()
            choosegate_B = F.sigmoid(choosegate_B).round()
            
            choosegate_1 = choosegate_A*choosegate_switch_1
            choosegate_2 = (1-choosegate_A)*choosegate_switch_1
            choosegate_3 = choosegate_B*(1-choosegate_switch_1)
            choosegate_4 = (1-choosegate_B)*(1-choosegate_switch_1)
            
            cellgate_1 = stretchgate*F.tanh(cellgate )       
            cellgate_2 = stretchgate*F.sigmoid(cellgate)        
            cellgate_3 = stretchgate*F.relu(cellgate)  
            cellgate_4 = stretchgate*F.tanhshrink(cellgate)
            
            cellgate = ((choosegate_1 * cellgate_1) + (choosegate_2 * cellgate_2) + (choosegate_3 * cellgate_3)  + (choosegate_4 * cellgate_4))
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
            cy = (forgetgate * cx) + ingate*cellgate
            hy = outgate * F.tanh(cy)
            return hy, cy
#####################################################################################################################################
if setting == 4: #CREATES AN ADAPTIVE ACTIVATION FUNCTION 
    print('confirm setting 4')
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        print('epoch#: ', epoch)
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh) 
        hx, cx = hidden
    
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate, choosegate_1, choosegate_2, choosegate_3, choosegate_4,choosegate_5 = gates.chunk(9, 1) 
        
        if setting_4_control == True:
#CONTROL            TO ILLUSTRATE THAT THE BENEFIT IS NOT ACHIEVED BY SIMPLY HAVING EXTRA LAYERS OR SOFTSIGN ACTIVATION FUNCTION
            choosegate_1 = F.tanhshrink(choosegate_1)            
            choosegate_2 = F.tanhshrink(choosegate_2)            
            choosegate_3 = F.tanhshrink(choosegate_3)      
            choosegate_4 = F.tanhshrink(choosegate_4) 
            choosegate_5 = F.tanhshrink(choosegate_5)        
            cellgate = choosegate_1 + choosegate_2 + choosegate_3 + choosegate_4 + choosegate_5 + cellgate
        else: 
#EXPERIMENTAL
            choosegate_1 = F.tanhshrink(choosegate_1) #+ 1e-4*(0 - F.tanhshrink(choosegate_1))           #height on y axis       (pushed to 0)
            choosegate_2 = F.tanhshrink(choosegate_2) #+ 1e-4*(1 - F.tanhshrink(choosegate_2))           #vertical length        (pushed to 1)
            choosegate_3 = F.tanhshrink(choosegate_3) #+ 1e-4*(1 - F.tanhshrink(choosegate_3))           #slope                  (pushed to 1)
            choosegate_4 = F.tanhshrink(choosegate_4) #+ 1e-4*(0 - F.tanhshrink(choosegate_4))           #position on x axis     (pushed to 0)
            choosegate_5 = F.tanhshrink(choosegate_5) #+ 1e-4*(0 - F.tanhshrink(choosegate_5))           #rectification          (pushed to 0)
            #this nudges each value towards a normal tanh conformation    
            cellgate = (choosegate_1 + choosegate_2*(F.tanh((choosegate_3*((choosegate_5*(torch.abs(cellgate)))+cellgate))+choosegate_4)))
            #this allows each to pull and stretch the softsign activation function 
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        outgate = F.sigmoid(outgate)
        
        ingate_x_cellgate = ingate*(cellgate) 
        cy = (forgetgate * cx) + ingate_x_cellgate
#        adaptgate_1 = F.sigmoid(adaptgate).round()
#        adaptgate_2 = (1-adaptgate_1)
#        hy = outgate * ((F.tanh(cy)*adaptgate_1)+(F.tanhshrink(cy)*adaptgate_2))
        hy = outgate * F.tanh(cy)
#        print(str(choosegate_1[1][1]),'+', str(choosegate_2[1][1]))
#        print(cellgate[1][1])
        return hy, cy

#######################################################################################################################################
if setting == 5: #RECURRENT LOOP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP
    def LSTMCell(input, hidden, w_ih, w_hh, w_2, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        
        
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
#        print(cellgate)
#        print(w_2)
        a =  cellgate * w_2
#        a =  cellgate * torch.transpose(w_2,0,1)
#        print(a)
#        print(cellgate)
        cellgate = cellgate + a
        outgate = F.sigmoid(outgate)
    
        cy = (forgetgate * cx) + (ingate * F.tanh(cellgate))
        hy = outgate * F.tanh(cy)
    
        return hy, cy
#######################################################################################################################################

if setting == 6: #simpler version of setting 4
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        
        
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        #MAYBE APPLY DENSITY CLUSTERING ANALYSIS AND DIMENSIONALITY REDUCTION HERE ON UPGATE AND STRETCHGATE
    
        if setting_6_version == 1:
        #STRETCH ALONE 
            ingate, forgetgate, cellgate, outgate, stretchgate = gates.chunk(5, 1)         
            stretchgate = F.tanhshrink(stretchgate)
            cellgate = F.tanh(stretchgate*cellgate)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * (F.tanh(cy))
            
        #TANH + RELU + STRETCH              
        if setting_6_version == 2:
            ingate, forgetgate, cellgate, outgate, stretchgate, choosegate = gates.chunk(6, 1)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
            
            stretchgate = F.tanhshrink(stretchgate)
            choosegate = F.sigmoid(choosegate).round()    
            cellgate = cellgate * stretchgate
            cellgate_tanh = (1-choosegate)*(F.tanh(cellgate))
            cellgate_relu = choosegate*(F.relu(cellgate))
            
            cellgate = cellgate_tanh + cellgate_relu
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)
            
        if setting_6_version == 3:
        #TANH VS RELU ALONE         
            ingate, forgetgate, cellgate, outgate, choosegate = gates.chunk(5, 1)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
            
            choosegate = F.sigmoid(choosegate).round()          
            cellgate_tanh =  F.tanh(cellgate)
            cellgate_relu =  F.relu(cellgate)
            
            cellgate = (cellgate_tanh*(choosegate)) + (cellgate_relu*(1-choosegate))
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)
            
        if setting_6_version == 4:    
            #Extra ingate 
            ingate, forgetgate, cellgate, outgate, stretchgate = gates.chunk(5, 1)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            outgate = F.sigmoid(outgate)
            ingate_2 = F.sigmoid(stretchgate)
        
            cellgate =  F.tanh(cellgate)
            cy = (forgetgate * cx) + (ingate * ingate_2 * cellgate)
            hy = outgate * F.tanh(cy)

        return hy, cy
##########################################################################################################################    
if setting == 6.5: 
    def LSTMCell(input, hidden, w_ih, w_hh, w_ih_2, b_ih=None,  b_ih_2 = None, b_hh=None):
        
        hx, cx = hidden
        #EXTRA LAYER        
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        outgate = F.sigmoid(outgate)
               
        cellgate =  F.tanh(cellgate)
        lay2 = F.linear(cellgate, w_ih_2, b_ih_2)
        cy = (forgetgate * cx) + (ingate * lay2)
        hy = outgate * F.tanh(cy)

        return hy, cy
#######################################################################################################################################
if setting == 7: #clustering activations SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP SKIP
        
    def LSTMCell(input, hidden, w_ih, w_hh,count, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        
        
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        
 #       if count == 0: 
#            cellmemory(:,:,count) = cellgate
        
        if preset == False: 
            ingate, forgetgate, cellgate, outgate, sensitivitygate_1,sensitivitygate_2,sensitivitygate_3 = gates.chunk(7, 1)
        
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            sensitivitygate_1 = F.relu(sensitivitygate_1)
            sensitivitygate_2 = F.sigmoid(sensitivitygate_2)
            sensitivitygate_3 = F.relu(sensitivitygate_3).round() 
    
            cellgatetemp = cellgate.unsqueeze_(2)
#            cellgatetemp.expand(:,:,torch.size(cellmemory,2))
              
            #with sensitivity gates
            sensitivitygate_1 = sensitivitygate.unsqueeze_(2)
#            sensitivitygate_1.expand(:,:,torch.size(cellmemory,2))
            sensitivitygate_2 = sensitivitygate.unsqueeze_(2)
#            sensitivitygate_2.expand(:,:,torch.size(cellmemory,2))
            sensitivitygate_3 = sensitivitygate.unsqueeze_(2)
#            sensitivitygate_3.expand(:,:,torch.size(cellmemory,2))
            direction = torch.pow(.0001*sensitivitygate_1,2)*sensitivitygate_2*(cellmemory-cellgatetemp)
            distance = torch.pow(.0001*sensitivtygate_1,2) + torch.pow((cellmemory-cellgate),(2*sensitivitygate_3))
        else:
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)    
            cellgatetemp = cellgate.unsqueeze_(2)
#            cellgatetemp.expand(:,:,torch.size(cellmemory,2))
            
            #with preset sensitivities 
#            a = #between 0 and infinity
#            b = #between 0 and 1        
#            c = #between 0 and infinity
            direction = torch.pow(.0001*a,2)*b*(cellmemory-cellgatetemp)
            distance = torch.pow(.0001*a,2) + torch.pow((cellmemory-cellgate),(2*c))        

 # direction = (a^2 * b * (x-z))
 # distance = (a^2 + (x-z)^(4*c))      
 # clustergrad = (a^2 * b * (x-z))/(a^2 + (x-z)^(4*c))          
            
        clustergrad = torch.sum((direction/distance),2)/torch.size(cellmemory,2)
        cellgate = F.tanh(cellgate + clustergrad)
#            cellmemory(:,:,count) = cellgate
        outgate = F.sigmoid(outgate)
    
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
    
        return hy, cy, cellmemory 
#######################################################################################################################################
if setting == 8: #clustering activations Method 3
        
    def LSTMCell(input, hidden, w_ih, w_hh,count, b_ih=None, b_hh=None):    
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
#       if effect == 'cellular':                
#             
#            ingate, forgetgate, position, outgate, sensitivitygate_1 = gates.chunk(5, 1)  
#            batch_size = position.size(0) 
#            a = F.relu(sensitivitygate_1)                               #determines the magnitude of the distance-weight correlation
#            a = torch.unsqueeze(a,2)
#            a = a.expand(-1, -1, 3*batch_size)                               #determines the magnitude of the distance-weight correlation
          
#            ingate = F.sigmoid(ingate)
#            forgetgate = F.sigmoid(forgetgate)  
            
 #           position = F.tanh(position)
            
 #           position = torch.cat((position,hx,cx),0)
 #           position = torch.unsqueeze(position,2)
 #           position = position.expand(-1,-1, 3*batch_size)
 #           distance = position - torch.transpose(position,dim0=0,dim1=2)
 #           distance = 1/(1+((distance[0:batch_size,:,:]**2)/(.01*a+.0000001)))   #take the distance from every other batch             
 #           weighted_avg = torch.sum((distance*position[0:batch_size,:,:]),2)  #and find the weighted average (smaller distance = larger weight)                           
 #           weighted_avg = weighted_avg/torch.sum(distance,2)
            
 #           outgate = F.sigmoid(outgate)   
 #           cy = (forgetgate * cx) + (ingate * weighted_avg)
 #           hy = outgate *  F.tanh(cy) 
 #           return hy, cy 
            
 #       if effect == 'network': 
                
 #       ingate, forgetgate, cellgate, outgate, sensitivitygate_1 = gates.chunk(5, 1)            
 #       batch_size = cellgate.size(0) 
 #       a = F.relu(sensitivitygate_1)                               #determines the magnitude of the distance-weight correlation
 #       a = torch.unsqueeze(a,2)
 #       a = a.expand(-1, -1, 3*batch_size)      
        
 #       ingate = F.sigmoid(ingate)
 #       forgetgate = F.sigmoid(forgetgate)          
 #       cellgate = F.tanh(cellgate)
 #       outgate = F.sigmoid(outgate)   
 #       cy = (forgetgate * cx) + (ingate * cellgate)
        
 #       position = F.tanh(cy)
        
 #       position = torch.cat((position,hx,cx),0)
 #       position = torch.unsqueeze(position,2)
 #       position = position.expand(-1,-1, 3*batch_size)
 #       distance = position - torch.transpose(position,dim0=0,dim1=2)
 #       distance = distance[0:batch_size,:,:] 
 #       distance = 1/(1+((distance**2)/(.01*a+.0000001)))   #take the distance from every other batch        
 #       weighted_avg = torch.sum((distance*position[0:batch_size,:,:]),2)  #and find the weighted average (smaller distance = larger weight)                           
 #       weighted_avg = weighted_avg/torch.sum(distance,2)

#        hy = outgate *  weighted_avg
        ingate, forgetgate, cellgate, outgate, sensitivitygate = gates.chunk(5, 1)            
        batch_size = cellgate.size(0) 
        a = F.relu(sensitivitygate)                               #determines the magnitude of the distance-weight correlation
        a = torch.unsqueeze(a,2)
        a = a.expand(-1, -1, 3*batch_size)       
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)          
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)   
        cy = (forgetgate * cx) + (ingate * cellgate)
        
        position = F.tanh(cy)
        position = torch.cat((position,hx,cx),0)
        position = torch.unsqueeze(position,2)
        position = position.expand(-1,-1, 3*batch_size)
        distance = (position - torch.transpose(position,dim0=0,dim1=2))**2
        distance = 1/(1+a*distance[0:batch_size,:,:])
        weighted_avg = torch.sum((distance*position[0:batch_size,:,:]),2)  #and find the weighted average (smaller distance = larger weight)                           
        weighted_avg = weighted_avg/torch.sum(distance,2)

        hy = outgate *  weighted_avg   
        return hy, cy 
#######################################################################################################################################
if setting == 9: #clustering activations Method 3
        
    def LSTMCell(input, hidden, w_ih, w_hh,count, b_ih=None, b_hh=None):    
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        if effect == 'cellular':                
            if preset == True:
                ingate, forgetgate, position, outgate = gates.chunk(4, 1) 
                batch_size = position.size(0) 
                #SET SENSITIVITY HERE
                a = .5 #between 0 and 1

            if preset == False: 
                ingate, forgetgate, position, outgate, sensitivitygate_1,sensitivitygate_2 = gates.chunk(6, 1)  
                batch_size = position.size(0) 
                a = F.sigmoid(sensitivitygate_1)                                #determines the magnitude of the distance-weight correlation          
                b = F.relu(sensitivitygate_2)                               #determines the magnitude of the distance-weight correlation
                
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)            
            newcell = torch.cuda.FloatTensor()
            position = F.tanh(position)
            position = torch.cat((position,hx,cx),0)
#            print('position',position.size())
            for i in range(batch_size):
                cellgatetemp = position[i,:]                                    #for each batch (for each neuron)
                torch.unsqueeze(cellgatetemp, 0)                                #dimensional correcton 
                cellgatetemp = cellgatetemp.expand(3*batch_size,-1)               #dimensional correction
 #               print('cellgate', cellgatetemp.size())     
 
                if preset == False: 
                    b = a[i,:]
                    torch.unsqueeze(b,0)
                    b = b.expand(3*batch_size,-1)  
                    distance = 1/(1+(((position - cellgatetemp)**2)/(.01*b+.0000001)))   #take the distance from every other batch 
                else:
                    distance = 1/((1/(b+1))+(((position - cellgatetemp)**2)/(.01*a+.0000001)))   #take the distance from every other batch        
                
                weighted_avg = torch.sum((distance*position),0)/torch.sum(distance,0) #and find the weighted average (smaller distance = larger weight)
                if i == 0:
                    newcell = F.tanh(torch.unsqueeze(weighted_avg,0))
                else:
                    newcell = F.tanh(torch.cat((newcell,torch.unsqueeze(weighted_avg,0)),0)) 
            outgate = F.sigmoid(outgate)   
            cy = (forgetgate * cx) + (ingate * newcell)
            position = F.tanh(cy)
            hy = outgate * F.tanh(cy)   
            return hy, cy 
            
        if effect == 'network': 
            if preset == True:
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)           
                #SET SENSITIVITY HERE
                a = .5 #between 0 and 1
                
            if preset == False: 
                ingate, forgetgate, cellgate, outgate, sensitivitygate_1, sensitivitygate_2 = gates.chunk(6, 1)            
                a = F.relu(sensitivitygate_1)                               #determines the magnitude of the distance-weight correlation
                b = F.relu(sensitivitygate_2)                               #determines the magnitude of the distance-weight correlation
                
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)         
            batch_size = cellgate.size(0)  
            cellgate = F.tanh(cellgate)
            newcell = torch.cuda.FloatTensor()
            outgate = F.sigmoid(outgate)   
            cy = (forgetgate * cx) + (ingate * cellgate)
            position = F.tanh(cy)
            position = torch.cat((position,hx,cx),0)
            for i in range(batch_size):
                cellgatetemp = position[i,:]                                    #for each batch (for each neuron)
                torch.unsqueeze(cellgatetemp, 0)                                #dimensional correcton 
                cellgatetemp = cellgatetemp.expand(3*batch_size,-1)               #dimensional correction             
                if preset == False: 
                    b = a[i,:]
                    torch.unsqueeze(b,0)
                    b = b.expand(3*batch_size,-1)  
                    distance = 1/(1+(((position - cellgatetemp)**2)/(.01*b+.0000001)))   #take the distance from every other batch 
                else:
                    distance = 1/((1/(b+1))+(((position - cellgatetemp)**2)/(.01*a+.0000001)))   #take the distance from every other batch        
          
                weighted_avg = torch.unsqueeze(torch.sum((distance*position),0)/torch.sum(distance,0),0) #and find the weighted average (smaller distance = larger weight)
                if i == 0:
                    newcell = weighted_avg
                else:
                    newcell = torch.cat((newcell,weighted_avg),0)
            hy = outgate *  newcell 
            return hy, cy 
#######################################################################################################################################
if setting == 10: #GENE EXPRESSION
    def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#        if input.is_cuda:
#            igates = F.linear(input, w_ih)
#            hgates = F.linear(hidden[0], w_hh)
#            state = fusedBackend.LSTMFused.apply
#            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
#        print('working')
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate, genegate = gates.chunk(5, 1)
        genegate = F.tanhshrink(genegate)
        genegate = F.tanh((genegate/((torch.mean(genegate,1)).unsqueeze(1))))*((torch.mean(genegate,1)).unsqueeze(1))
        a = torch.mean(genegate[:,0:3],1).unsqueeze(1)
        b = torch.mean(genegate[:,4:7],1).unsqueeze(1)
        c = torch.mean(genegate[:,8:11],1).unsqueeze(1)
        d = torch.mean(genegate[:,12:15],1).unsqueeze(1)
        e = torch.mean(genegate[:,16:19],1).unsqueeze(1)
        f = torch.mean(genegate[:,20:23],1).unsqueeze(1)
        g = torch.mean(genegate[:,24:27],1).unsqueeze(1)
        h = torch.mean(genegate[:,28:31],1).unsqueeze(1)
        i = torch.mean(genegate[:,32:35],1).unsqueeze(1)
        j = torch.mean(genegate[:,36:39],1).unsqueeze(1)
        k = torch.mean(genegate[:,40:43],1).unsqueeze(1)
        l = torch.mean(genegate[:,44:47],1).unsqueeze(1)
        m = torch.mean(genegate[:,48:51],1).unsqueeze(1)
        n = torch.mean(genegate[:,52:55],1).unsqueeze(1)
        o = torch.mean(genegate[:,56:59],1).unsqueeze(1)
        p = torch.mean(genegate[:,60:63],1).unsqueeze(1)
        q = torch.mean(genegate[:,64:67],1).unsqueeze(1)
        r = torch.mean(genegate[:,68:71],1).unsqueeze(1)
        s = torch.mean(genegate[:,72:75],1).unsqueeze(1)     
        t = torch.mean(genegate[:,76:79],1).unsqueeze(1)
        u = torch.mean(genegate[:,80:83],1).unsqueeze(1)
        v = torch.mean(genegate[:,84:87],1).unsqueeze(1)
        w = torch.mean(genegate[:,88:91],1).unsqueeze(1)
        x = torch.mean(genegate[:,92:95],1).unsqueeze(1)
        y = torch.mean(genegate[:,96:99],1).unsqueeze(1)
        z = torch.mean(genegate[:,100:150],1).unsqueeze(1)
    
        ingate = d+ c*F.sigmoid(a+(b*ingate**u))
        forgetgate = h+g*F.sigmoid(f+(e*forgetgate**v))
        cellgate = l+k*F.tanh(j+(i*cellgate**w))
        outgate = t+ s*F.sigmoid(r+(q*outgate**x))
    
        cy = y*((forgetgate * cx) + (ingate * cellgate))
        hy = z*(outgate * (p + o*F.tanh(n+(m*cy))))
    
        return hy, cy
#######################################################################################################################################
if setting == 6: 
    def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

#    if input.is_cuda:
#        gi = F.linear(input, w_ih)
#        gh = F.linear(hidden, w_hh)
#        state = fusedBackend.GRUFused.apply
#        return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)

        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n, i_c = gi.chunk(4, 1)
        h_r, h_i, h_n, h_c = gh.chunk(4, 1)
    
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        choosegate = F.sigmoid(i_c + h_c)
        newgate_tan = F.tanh(i_n + resetgate * h_n)*choosegate
        newgate_rel = F.relu(i_n + resetgate * h_n)*(1-choosegate)
        newgate = newgate_tan + newgate_rel
        hy = newgate + inputgate * (hidden - newgate)
    
        return hy
else:
    def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    
#        if input.is_cuda:
#            gi = F.linear(input, w_ih)
 #           gh = F.linear(hidden, w_hh)
 #           state = fusedBackend.GRUFused.apply
 #           return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
    
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
    
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
    
        return hy



def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)
    return fac


def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None):

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    elif mode == 'GRU':
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


class CudnnRNN(NestedIOFunction):

    def __init__(self, mode, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout=0, train=True, bidirectional=False,
                 batch_sizes=None, dropout_state=None, flat_weight=None):
        super(CudnnRNN, self).__init__()
        if dropout_state is None:
            dropout_state = {}
        self.mode = cudnn.rnn.get_cudnn_mode(mode)
        self.input_mode = cudnn.CUDNN_LINEAR_INPUT
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.train = train
        self.bidirectional = 1 if bidirectional else 0
        self.num_directions = 2 if bidirectional else 1
        self.batch_sizes = batch_sizes
        self.dropout_seed = torch.IntTensor(1).random_()[0]
        self.dropout_state = dropout_state
        self.weight_buf = flat_weight
        if flat_weight is None:
            warnings.warn("RNN module weights are not part of single contiguous "
                          "chunk of memory. This means they need to be compacted "
                          "at every call, possibly greatly increasing memory usage. "
                          "To compact weights again call flatten_parameters().", stacklevel=5)

    def forward_extended(self, input, weight, hx):
        assert cudnn.is_acceptable(input)
        # TODO: raise a warning if weight_data_ptr is None

        output = input.new()

        if torch.is_tensor(hx):
            hy = hx.new()
        else:
            hy = tuple(h.new() for h in hx)

        cudnn.rnn.forward(self, input, hx, weight, output, hy)

        self.save_for_backward(input, hx, weight, output)
        return output, hy

    def backward_extended(self, grad_output, grad_hy):
        input, hx, weight, output = self.saved_tensors
        input = input.contiguous()

        grad_input, grad_weight, grad_hx = None, None, None

        assert cudnn.is_acceptable(input)

        grad_input = input.new()
        if torch.is_tensor(hx):
            grad_hx = input.new()
        else:
            grad_hx = tuple(h.new() for h in hx)

        if self.retain_variables:
            self._reserve_clone = self.reserve.clone()

        cudnn.rnn.backward_grad(
            self,
            input,
            hx,
            weight,
            output,
            grad_output,
            grad_hy,
            grad_input,
            grad_hx)

        if any(self.needs_input_grad[1:]):
            grad_weight = [tuple(w.new().resize_as_(w) for w in layer_weight) for layer_weight in weight]
            cudnn.rnn.backward_weight(
                self,
                input,
                hx,
                output,
                weight,
                grad_weight)
        else:
            grad_weight = [(None,) * len(layer_weight) for layer_weight in weight]

        if self.retain_variables:
            self.reserve = self._reserve_clone
            del self._reserve_clone

        return grad_input, grad_weight, grad_hx


def hack_onnx_rnn(fargs, output, args, kwargs):
    input, all_weights, hx = fargs
    output_tensors = tuple(v.data for v in _iter_variables(output))
    flat_weights = tuple(_iter_variables(all_weights))
    flat_hx = tuple(_iter_variables(hx))

    class RNNSymbolic(Function):
        @staticmethod
        def symbolic(g, *fargs):
            # NOTE: fargs contains Variable inputs (input + weight + hidden)
            # NOTE: args/kwargs contain RNN parameters
            raise RuntimeError("hack_onnx_rnn NYI")

        @staticmethod
        def forward(ctx, *fargs):
            return output_tensors

        @staticmethod
        def backward(ctx, *gargs, **gkwargs):
            raise RuntimeError("FIXME: Traced RNNs don't support backward")

    flat_output = RNNSymbolic.apply(*((input,) + flat_weights + flat_hx))
    return _unflatten(flat_output, output)


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        if cudnn.is_acceptable(input.data):
            func = CudnnRNN(*args, **kwargs)
        else:
            func = AutogradRNN(*args, **kwargs)

        # Hack for the tracer that allows us to represent RNNs as single
        # nodes and export them to ONNX in this form
        if torch._C._jit_is_tracing(input):
            assert not fkwargs
            output = func(input, *fargs)
            return hack_onnx_rnn((input,) + fargs, output, args, kwargs)
        else:
            return func(input, *fargs, **fkwargs)

    return forward
