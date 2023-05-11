

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

l <- 100L 
input <- layer_input(shape = l)

gen <- input%>%
  layer_dense(units = 128 ,input_shape = l) %>% 
  layer_activation_relu( )%>% 
  layer_dense(units = 256) %>%  
  layer_activation_relu( )%>% 
  layer_dense(units = 340) %>%  
  layer_activation_relu( )%>% 
  layer_lambda( f = Lsn  )   

generator <- keras_model(input,gen)
generator 


generator %>% compile(optimizer = optimizer_adamax(learning_rate = 0.0001),
                      loss = 'cosine_similarity' )

batch <- 5

gloss <- NULL
g_coor_p <- NULL
g_coor_s <- NULL

for (i in 1:2000) {
  noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
  real <- normal_tissure[,6:10]
  real1 <- t(real) 
  
  gloss[i]  <- generator%>%train_on_batch( noise,real1)
  
  
  fake <- predict_on_batch(generator,noise)
  real <- t(normal_tissure)
  data_t <- t(rbind(fake,real))
  x_corr <- rcorr(data_t,type = 'pearson')
  r <- x_corr$r
  r <- as.data.frame(r)[1:5,11:15]
  g_coor_p[i] <- mean(as.matrix(r))
  
  data_rank <- apply(data_t, 2, rank)
  x_corr <- rcorr(data_rank,type ='spearman')
  r <- x_corr$r
  r <- as.data.frame(r)[1:5,11:15]
  g_coor_s[i] <- mean(as.matrix(r))
  
  print(i)
  #   print( d_loss_fake)
  #  print( d_loss_real)
  print(  gloss[i])
  print(g_coor_p[i])
}
 
########### generator #########
shape <- length( rownames(normal_single_cell)) 

input0 <- layer_input(shape = shape)

dis <- input0%>% 
  tfa$layers$SpectralNormalization(layer_dense(units = 256,activation = 'relu' ))()%>%
  layer_dropout(rate = 0.2)%>%   
  tfa$layers$SpectralNormalization(layer_dense(units = 128,activation = 'relu' ))()%>%
  layer_dropout(rate = 0.2)%>%  
  tfa$layers$SpectralNormalization(layer_dense(units = 64,activation = 'relu' ))()%>%
  layer_dropout(rate = 0.2)%>%  
  tfa$layers$SpectralNormalization(layer_dense(units = 32,activation = 'relu' ))()%>%
  layer_dense(units = 10 )%>%
  k_l2_normalize()  

discriminator <- keras_model(input0,dis)
discriminator 

discriminator %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.0001)  )

generator %>% compile(optimizer = optimizer_rmsprop(learning_rate = 0.0001) )



##### GNA## 
batch <- 20
dloss <- NULL
gloss <- NULL

g_sham_p <- NULL
g_mi2_p <- NULL
g_mi7_p <- NULL

real <- t(normal_tissure[,c(1:10,26:30)])  

sham_res <- array(runif(5,1,1),dim =c(5,1))
MI2_res <- array(runif(5,2,2),dim =c(5,1)) 
MI7_res <- array(runif(5,3,3),dim =c(5,1))
fake_res <- array(runif(batch,4,4),dim =c(batch,1))

lab <- rbind(sham_res,MI2_res,MI7_res,fake_res)
 
pre_sham <- rbind(sham_res,MI2_res,MI7_res, array(runif(batch,1,1),dim =c(batch,1)))

pre_mi2 <- rbind(sham_res,MI2_res, MI7_res,array(runif(batch,2,2),dim =c(batch,1)))
 
pre_mi7 <- rbind(sham_res,MI2_res, MI7_res,array(runif(batch,3,3),dim =c(batch,1)))

for (i in 1:2000) {
   ## class  
  for (j in 1:5) {
    
    noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
    with(tf$GradientTape()%as% tape, {
      fake_sham <- generator(noise,training = F)
      
      matrix_log <- discriminator(rbind(real,fake_sham),training = T)
      
      #d_loss<-tfa$losses$triplet_semihard_loss(lab,rbind(real_log,fake_log), distance_metric = "squared-L2",margin = 0.1)
      d_loss<-tfa$losses$triplet_semihard_loss(lab,matrix_log,distance_metric = "squared-L2",margin = 0.1)
      d_grad <- tape$gradient(d_loss,discriminator$trainable_variables)
      discriminator$optimizer$apply_gradients(zip_lists(d_grad,discriminator$trainable_variables))
    })
    
  }
    gc() 
    noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
    ##generator sham
    with(tf$GradientTape()%as% tape, {
      fake_sham <- generator(noise,training = T)
      matrix_log <- discriminator(rbind(real,fake_sham),training = F)

    #g_loss <- tfa$losses$triplet_semihard_loss(pre_lab,rbind(real_log,fake_log), distance_metric = "squared-L2",margin = 0.1)
    g_loss<-tfa$losses$triplet_semihard_loss(pre_sham,matrix_log,distance_metric = "squared-L2",margin = 0.1)
    
    g_grad <- tape$gradient(g_loss,generator$trainable_variables)
    generator$optimizer$apply_gradients(zip_lists(g_grad,generator$trainable_variables))
  })
  
   dloss[i]  <- as.array( d_loss)
   gloss[i]  <- as.array( g_loss)
  
   fake <- predict_on_batch(generator,noise)
   data_t <- t(rbind(fake,real))
   x_corr <- rcorr(data_t,type = 'pearson')
   r <- x_corr$r
   
   r_sham <- as.data.frame(r)[1:20,21:25]
   g_sham_p[i] <- mean(as.matrix(r_sham))
   
   r_MI2 <- as.data.frame(r)[1:20,26:30]
   g_mi2_p[i] <- mean(as.matrix(r_MI2))
   
   r_MI7 <- as.data.frame(r)[1:20,31:35]
   g_mi7_p[i] <- mean(as.matrix(r_MI7))
   
  print(i)
  #   print( d_loss_fake)
  #  print( d_loss_real)
  print(  dloss[i])
  print(  g_sham_p[i])
}


for (i in 1:6000) {
 
  noise <- k_random_normal(c(batch,l),mean = 0,stddev = 1)
  ##generator sham
  with(tf$GradientTape()%as% tape, {
    fake_sham <- generator_sham(noise,training = T)
    fake_mi2 <- generator_mi2(noise,training = F)    
    matrix_log <- discriminator(rbind(real,fake_sham,fake_mi2),training = F)
    
    #g_loss <- tfa$losses$triplet_semihard_loss(pre_lab,rbind(real_log,fake_log), distance_metric = "squared-L2",margin = 0.1)
    g_loss<-tfa$losses$triplet_hard_loss(pre_sham,matrix_log,distance_metric = "squared-L2",margin = 0.001)
    
    g_grad <- tape$gradient(g_loss,generator_sham$trainable_variables)
    generator_sham$optimizer$apply_gradients(zip_lists(g_grad,generator_sham$trainable_variables))
  })
  ##generator mi2
  with(tf$GradientTape()%as% tape, {
    fake_sham <- generator_sham(noise,training = F)
    fake_mi2 <- generator_mi2(noise,training = T)    
    matrix_log <- discriminator(rbind(real,fake_sham,fake_mi2),training = F)
    
    #g_loss <- tfa$losses$triplet_semihard_loss(pre_lab,rbind(real_log,fake_log), distance_metric = "squared-L2",margin = 0.1)
    g_loss<-tfa$losses$triplet_hard_loss(pre_mi2,matrix_log,distance_metric = "squared-L2",margin = 0.001)
    
    g_grad <- tape$gradient(g_loss,generator_mi2$trainable_variables)
    generator_mi2$optimizer$apply_gradients(zip_lists(g_grad,generator_mi2$trainable_variables))
  })
  
  dloss[i]  <- as.array( d_loss)
  gloss[i]  <- as.array( g_loss)
  
  
  print(i)
  #   print( d_loss_fake)
  #  print( d_loss_real)
  print(  dloss[i])
  print(  gloss[i])
}

data_cosin <- cbind(gloss,dloss,g_sham_p,g_mi2_p,g_mi7_p)


fake_sham <- predict_on_batch(generator,noise) 
group_fake <- c(group,rep('fake_sham',30))

data_t <-  t(normal_tissure) 
data_t <-as.data.frame(t( rbind(data_t,fake_sham )))
# data_t <- myscale(data_t)
group_fake <- c(group,rep('sham',30) )

pca <- PCA(t(data_t), scale.unit = TRUE, ncp = 5, graph = T)

fviz_pca_ind(pca,pointsize = 4, pointshape = 20, 
             labelsize = 4,  repel=T,  fill.ind = group_fake, col.ind = group_fake,
             # palette=c('#F16300','#68B195' ),
             addEllipses = F,
             ellipse.level=0.97,
             mean.point=F,
             #  legend.title = "Groups",
             ggtheme = theme_classic()) 

data_t <-  t(normal_tissure) 
emb <- predict_on_batch(discriminator,rbind(data_t,fake_sham ))

pca <- PCA(emb, scale.unit = TRUE, ncp = 5, graph = T)

fviz_pca_ind(pca,pointsize = 4, pointshape = 20, 
             labelsize = 4,  repel=T,  fill.ind = group_fake, col.ind = group_fake,
             # palette=c('#F16300','#68B195' ),
             addEllipses = F,
             ellipse.level=0.97,
             mean.point=F,
             #  legend.title = "Groups",
             ggtheme = theme_classic()) 





