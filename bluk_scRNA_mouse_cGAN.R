

library(factoextra)
library(FactoMineR)
library(keras)
library(tensorflow)
library(tfdatasets)
library(ggplot2)
library(reticulate)
library(Hmisc)

tfa <- import('tensorflow_addons') 


RNA <- read.csv('~/Bio_project/snRNA/mouse/GSE183272.csv' ,stringsAsFactors = F,row.names = 1)


single_cell <-  read.csv('~/Bio_project/snRNA/mouse/single_cell.csv', row.names =1,stringsAsFactors = F)

count <- as.data.frame(apply(single_cell,2,sum))
colnames(count) <-'count'

gene <- Reduce(intersect,list(rownames(RNA),rownames(single_cell) ))
#exp <- exp[which(rownames(exp) %in% gene),]

feature <- c('Col8a1','Col3a1','Col1a1','Dcn','Gsn',
             'Ryr2','Ttn','Myh6','Hif1a', 'Vegfa',
             'Adgre1','Ptprc','Apoe','Top2a','Mki67',
             'Pdgfd','Fabp4', 'Cd36', 
             'Mylk','Rgs5','Cald1','Myh11',
             'Adgrd1','Vwf','Pecam1',
             'Msln','C3', ## EA
             'Skap2','Fyb')

mark <- read.csv('~/Bio_project/snRNA/mouse/mark.csv',stringsAsFactors = F )
mark <- mark[which(mark$avg_log2FC > 2),]
feature <- unique(c(feature, mark$gene))

seclect  <- Reduce(intersect,list(feature,gene))
## 
normal_tissure <- RNA[which(rownames(RNA) %in% seclect),]
normal_single_cell <- single_cell[which(rownames(single_cell) %in% seclect),]

## tpm tissure 
normal_tissure$Gene <- rownames(normal_tissure)  
normal_tissure$Gene <- factor(normal_tissure$Gene,levels = rownames(normal_single_cell))
normal_tissure <- normal_tissure[order(normal_tissure$Gene  ),]
normal_tissure <- normal_tissure[,-31]

data_t <-  t(normal_tissure) 
pca <- PCA(data_t , scale.unit = TRUE, ncp = 5, graph = T)


group <- c(rep('sham_d2',5),rep('MI_d2',5),
           rep('sham_d4',5),rep('MI_d4',5),
           rep('sham_d7',5),rep('MI_d7',5))

fviz_pca_ind(pca,pointsize = 4, pointshape = 20, 
             labelsize = 4,  repel=T,# 
             fill.ind  = group, col.ind = group,
             #  palette=c('#F16300','#68B195' ),
             addEllipses = F,
             ellipse.level=0.97,
             mean.point=F,
             #  legend.title = "Groups",
             ggtheme = theme_classic()) 



input1 <- as_tensor(   as.matrix(normal_single_cell)  , dtype = 'float32')
input1 <- k_expand_dims(input1,axis = 1)


input2 <- as_tensor(   t(count)  , dtype = 'float32')
#input3 <- k_expand_dims(input3,axis = 1) 

### tpm to tpm 
Lsn <- function(output){  
  my_variable <- k_repeat(output,length(normal_single_cell[,1]))
  res <-  my_variable*input1 ## count_matrix
  my_sum <-  k_sum(res,axis = -1) 
  
  count <- output*input2  ## count sum
  my_variable <- k_repeat(count,length(normal_single_cell[,1]))
  my_variable <- k_sum(my_variable,axis = -1) 
  count <- 10000/my_variable
  
  return(count * my_sum)
} 

### gen####
l <- 100L

input_g <- layer_input(shape = l)
label1 <- layer_input(shape=list(1))

label_embedding <- label1 %>%  
  layer_embedding(input_dim = 3,output_dim =  l)  %>%   
  layer_flatten() 

model_input <- layer_concatenate(list(input_g,label_embedding) )

gen <- keras_model_sequential()
gen <- model_input %>% 
  layer_dense(units = 256) %>%  
  layer_activation_relu( )%>% 
  layer_dense(units = 512) %>%  
  layer_activation_relu( )%>% 
  layer_dense(units = 800) %>%  
  layer_activation_relu( )%>% 
  layer_lambda( f = Lsn  )    


generator <- keras_model(list(input_g,label1),gen)
generator


### dis 
shape <- length( rownames(normal_single_cell))

input_d <- layer_input(shape = shape)
label2 <- layer_input(shape=list(1))

label_embedding2 <- label2 %>%  
  layer_embedding(input_dim = 3,output_dim =  shape)  %>%   
  layer_flatten() 

model_input2 <- layer_concatenate(list(input_d,label_embedding2))


dis <- keras_model_sequential()
dis <-  model_input2 %>% 
  layer_dense(units = 256 )%>%
  layer_activation_leaky_relu(alpha = 0.2 )%>%
  layer_dense(units = 128  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 64 )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 16  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 8  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_flatten()%>%  
  layer_dense(units = 1 ,activation = 'sigmoid')

discriminator <- keras_model(list(input_d,label2),dis)
discriminator 
##### GNA## 

discriminator %>% compile(optimizer = optimizer_adamax(learning_rate = 0.0001),
                          loss = 'binary_crossentropy'   )

discriminator$trainable

freeze_weights(discriminator)
discriminator$trainable


input <- layer_input(shape = list(l))
label <- layer_input(shape=list(1)) 
img <- generator(list(input,label))
validity <- discriminator(list(img,label))

GAN <- keras_model(list(input,label),validity)

GAN %>% compile(optimizer = optimizer_adamax(learning_rate = 0.0001),
                loss = 'binary_crossentropy'   )

##fit 
batch <- 15

dloss <- NULL
gloss <- NULL
g_coor_p <- NULL
g_coor_s <- NULL



sham_res <- array(runif(5,0,0),dim =c(5,1))
MI2_res <- array(runif(5,1,1),dim =c(5,1)) 
MI7_res <- array(runif(5,2,2),dim =c(5,1))

## 
for (i in 1:200) {
  
  unfreeze_weights(discriminator)
  
  for (j in 1:5) {
    real <- normal_tissure[,c(1:10,26:30)]
    real <- t(real) 
    real_label <- rbind(sham_res,MI2_res,MI7_res)
    real_res <- array(runif(batch,1,1),dim =c(batch,1)) 
    d_loss_real <- discriminator%>%train_on_batch(list(real,real_label),real_res)
    
    noise <- matrix(rnorm(batch*l),nrow = batch,ncol = l)
    noise_label <- sample(0:2,batch,replace = T) %>% matrix(ncol = 1)
    fake <- predict_on_batch(generator,list(noise,noise_label) )
    fake_res <- array(runif(batch,0,0),dim =c(batch,1)) 
    d_loss_fake <- discriminator%>%train_on_batch(list(fake,noise_label),fake_res)
    
    dloss_batch <- d_loss_real +d_loss_fake 
    
  }
  dloss[i] <-  dloss_batch
  gc()
  
  freeze_weights(discriminator)
  noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
  noise_label <- sample(0:2,batch,replace = T) %>% matrix(ncol = 1) 
  real_res <- array(runif(batch,1,1),dim =c(batch,1)) 
  
  gloss[i]  <- GAN%>%train_on_batch(list(noise,noise_label),real_res)
  
  
  fake <- predict_on_batch(generator,list(noise,noise_label))
  data_t <- t(rbind(fake,real))
  x_corr <- rcorr(data_t,type = 'pearson')
  r <- x_corr$r
  r <- as.data.frame(r)[1:15,16:30]
  g_coor_p[i] <- mean(as.matrix(r))
  
  data_rank <- apply(data_t, 2, rank)
  x_corr <- rcorr(data_rank,type ='spearman')
  r <- x_corr$r
  r <- as.data.frame(r)[1:15,16:30]
  g_coor_s[i] <- mean(as.matrix(r))
  
  print(i)
  #   print( d_loss_fake)
  #  print( d_loss_real)
  print(  dloss[i])
  print( g_coor_p[i] )
}


data_cosin <- cbind(gloss,dloss,g_coor_s,g_coor_p)

write.csv(data_cosin,file = 'relation.csv')
save_model_weights_hdf5(generator,"generator_cgan_2000.h5")
save_model_weights_hdf5(discriminator,"discriminator_cgan_2000.h5")


load_model_weights_hdf5(generator,"cgan_gan_weight.h5")
load_model_weights_hdf5(discriminator,"cgan_dis_weight.h5")

noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
noise_label1 <- sample(0:0,batch,replace = T) %>% matrix(ncol = 1)  
noise_label2 <- sample(1:1,batch,replace = T) %>% matrix(ncol = 1)  
noise_label3 <- sample(2:2,batch,replace = T) %>% matrix(ncol = 1)  

fake1 <- predict_on_batch(generator,list(noise,noise_label1))
fake2 <- predict_on_batch(generator,list(noise,noise_label2))
fake3 <- predict_on_batch(generator,list(noise,noise_label3))
data_t <- rbind(t(normal_tissure),fake1,fake2,fake3)
pca <- PCA(data_t, scale.unit = TRUE, ncp = 5, graph = T)
group_fake <- c(group,rep('fake',15),rep('fake2',15),rep('fake3',15))

fviz_pca_ind(pca,pointsize = 4, pointshape = 20, 
             labelsize = 4,  repel=T, fill.ind  = group_fake, col.ind = group_fake,
             #palette=c('#F16300','#68B195' ),
             addEllipses = F,
             ellipse.level=0.97,
             mean.point=F,
             #  legend.title = "Groups",
             ggtheme = theme_classic()) 


pca <- PCA( predict_on_batch(classification,data_t), scale.unit = TRUE, ncp = 5, graph = T)

fviz_pca_ind(pca,pointsize = 4, pointshape = 20, 
             labelsize = 4,  repel=T, fill.ind  = group_fake, col.ind = group_fake,
             #palette=c('#F16300','#68B195' ),
             addEllipses = F,
             ellipse.level=0.97,
             mean.point=F,
             #  legend.title = "Groups",
             ggtheme = theme_classic()) 



fake <- predict_on_batch(generator,list(noise,noise_label))
data_t <- t(rbind(fake,real))
x_corr <- rcorr(data_t,type = 'pearson')
r <- x_corr$r
r <- as.data.frame(r)[1:15,16:30]




## dis without lab#####
##failed for diffined dis fake res

shape <- length( rownames(normal_single_cell))

input_d <- layer_input(shape = shape) 

dis <- keras_model_sequential()
dis <-  input_d %>%  
  layer_dense(units = 256 )%>%
  layer_activation_leaky_relu(alpha = 0.2 )%>%
  layer_dense(units = 128  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 64 )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 16  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_dense(units = 8  )%>%
  layer_activation_leaky_relu(alpha = 0.2)%>%
  layer_flatten()%>%  
  layer_dense(units = 3 ,activation = 'softmax')


discriminator <- keras_model(input_d,dis)
discriminator 
##### GNA## 
 

discriminator %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.0001),
                          loss = 'categorical_crossentropy'  )

discriminator$trainable

freeze_weights(discriminator)
discriminator$trainable


input <- layer_input(shape = list(l))
label <- layer_input(shape=list(1)) 
img <- generator(list(input,label))
validity <- discriminator(img)

GAN <- keras_model(list(input,label),validity)

GAN %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.0001),
                loss = 'categorical_crossentropy'  )


for (i in 1:2000) {
  
  unfreeze_weights(discriminator)
  
  for (j in 1:5) {
    real <- normal_tissure[,c(1:10,26:30)]
    real <- t(real) 
    real_label <- rbind(sham_res,MI2_res,MI7_res)
    real_res <- to_categorical(real_label,3)
    d_loss_real <- discriminator%>%train_on_batch(real,real_res)
    
    noise <- matrix(rnorm(batch*l),nrow = batch,ncol = l)
    noise_label <- sample(0:2,batch,replace = T) %>% matrix(ncol = 1)
    fake <- predict_on_batch(generator,list(noise,noise_label) )
    fake_res <-  (1-to_categorical(noise_label,3))/2
    d_loss_fake <- discriminator%>%train_on_batch(fake,fake_res)
    
    dloss_batch <- d_loss_real +d_loss_fake 
    
  }
  dloss[i] <-  dloss_batch
  gc()
  
  freeze_weights(discriminator)
  noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
  noise_label <- sample(0:2,batch,replace = T) %>% matrix(ncol = 1) 
  fake_res <- to_categorical(noise_label,3)
  gloss[i]  <- GAN%>%train_on_batch(list(noise,noise_label),fake_res)
  
  
  fake <- predict_on_batch(generator,list(noise,noise_label))
  data_t <- t(rbind(fake,real))
  x_corr <- rcorr(data_t,type = 'pearson')
  r <- x_corr$r
  r <- as.data.frame(r)[1:15,16:30]
  g_coor_p[i] <- mean(as.matrix(r))
  
  data_rank <- apply(data_t, 2, rank)
  x_corr <- rcorr(data_rank,type ='spearman')
  r <- x_corr$r
  r <- as.data.frame(r)[1:15,16:30]
  g_coor_s[i] <- mean(as.matrix(r))
  
  print(i)
  #   print( d_loss_fake)
  #  print( d_loss_real)
  print(  dloss[i])
  print( g_coor_p[i] )
}


