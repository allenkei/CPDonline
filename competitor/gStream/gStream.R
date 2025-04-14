library(gStream)
library(reticulate)



np <- import("numpy")
data_use <- "s1"     ### "s1","s2","s3","s4","s5"
dim_d <- 10          ### 10, 20
quant <- 0.95        ### 0.95, 0.975
quant_name <- 95     ### 95, 975

data <- np$load(sprintf("data/data_%s_d%d.npz", data_use, dim_d))
data_noCP <- np$load(sprintf("data/data_%s_d%d_C.npz", data_use, dim_d))


set.seed(1)
Delta <- 50 # CP
N0 = 30 # history
K = 7
L = 30

##############
# WITHOUT CP #
##############


# y_list <- data_noCP$f[['y_list']]
# holder_ori<-holder_wei<-holder_max<-holder_gen<-matrix(0, nrow=dim(y_list)[1],ncol=dim(y_list)[1])
# 
# for(i in 1:dim(y_list)[1]) {
#   
#   one_seq <- y_list[i,,]
#   distM1 <- as.matrix(dist(one_seq)); diag(distM1) <- max(distM1)+100
#   
#   r1 = gstream(distM1, L, N0, K, statistics="all", n0=0.3*L, n1=0.7*L,
#                ARL=20000, alpha=0, skew.corr=FALSE, asymp=FALSE) 
#   
#   holder_ori[i,(N0+1):100] <- r1$scanZ$ori
#   holder_wei[i,(N0+1):100] <- r1$scanZ$weighted
#   holder_gen[i,(N0+1):100] <- r1$scanZ$generalized
#   holder_max[i,(N0+1):100] <- r1$scanZ$max.type
#   
# }
# 
# 
# 
# write.csv(holder_ori, sprintf("data/threshold_%s_d%d_ori.csv", data_use, dim_d), row.names = F)
# write.csv(holder_wei, sprintf("data/threshold_%s_d%d_wei.csv", data_use, dim_d), row.names = F)
# write.csv(holder_gen, sprintf("data/threshold_%s_d%d_gen.csv", data_use, dim_d), row.names = F)
# write.csv(holder_max, sprintf("data/threshold_%s_d%d_max.csv", data_use, dim_d), row.names = F)
# 
# 
# quantile_ori <- quantile(apply(holder_ori, 1, max), quant)
# quantile_wei <- quantile(apply(holder_wei, 1, max), quant)
# quantile_gen <- quantile(apply(holder_gen, 1, max), quant)
# quantile_max <- quantile(apply(holder_max, 1, max), quant)




###########
# WITH CP #
###########


ori_thres <- as.matrix(read.csv(sprintf("data/threshold_%s_d%d_ori.csv", data_use, dim_d)))
wei_thres <- as.matrix(read.csv(sprintf("data/threshold_%s_d%d_wei.csv", data_use, dim_d)))
gen_thres <- as.matrix(read.csv(sprintf("data/threshold_%s_d%d_gen.csv", data_use, dim_d)))
max_thres <- as.matrix(read.csv(sprintf("data/threshold_%s_d%d_max.csv", data_use, dim_d)))


quantile_ori <- quantile(apply(ori_thres, 1, max), quant)
quantile_wei <- quantile(apply(wei_thres, 1, max), quant)
quantile_gen <- quantile(apply(gen_thres, 1, max), quant)
quantile_max <- quantile(apply(max_thres, 1, max), quant)





y_list <- data$f[['y_list']]
output_ori<-output_wei<-output_max<-output_gen<-numeric(dim(y_list)[1])


for(i in 1:dim(y_list)[1]) {
  
  one_seq <- y_list[i,,]
  distM1 <- as.matrix(dist(one_seq)); diag(distM1) <- max(distM1)+100
  
  r1 = gstream(distM1, L, N0, K, statistics="all", n0=0.3*L, n1=0.7*L,
               ARL=20000, alpha=0, skew.corr=FALSE, asymp=FALSE) 
  
  output_ori[i] <- which(r1$scanZ$ori > quantile_ori)[1] + N0
  output_wei[i] <- which(r1$scanZ$weighted > quantile_wei)[1] + N0
  output_gen[i] <- which(r1$scanZ$generalized > quantile_gen)[1] + N0
  output_max[i] <- which(r1$scanZ$max.type > quantile_max)[1] + N0
  
}


for(i in 1:dim(y_list)[1]){
  if(is.na(output_ori[i])) output_ori[i] <- 100
  if(is.na(output_wei[i])) output_wei[i] <- 100
  if(is.na(output_gen[i])) output_gen[i] <- 100
  if(is.na(output_max[i])) output_max[i] <- 100
}







eval <- function(output_t0, Delta, num_T) {
  
  N <- length(output_t0)
  
  delay_sum <- sum(output_t0[output_t0 >= Delta] - Delta)  # numerator
  delay_count <- sum(output_t0 >= Delta)  # denominator
  
  pfa_count <- sum(output_t0 < Delta)  # numerator
  pfn_count <- sum(output_t0 == num_T)  # numerator
  
  #print(output_t0[output_t0 >= Delta])
  delay <- delay_sum / delay_count
  pfa <- pfa_count / N
  pfn <- pfn_count / N
  
  return(c(delay, pfa, pfn))
}



result_ori <- eval(output_ori, Delta=50, num_T=100); result_ori
result_wei <- eval(output_wei, Delta=50, num_T=100); result_wei
result_gen <- eval(output_gen, Delta=50, num_T=100); result_gen
result_max <- eval(output_max, Delta=50, num_T=100); result_max



write.csv(result_ori, sprintf("result/%s/result_%s_d%d_ori_q%d.csv", 
                              data_use, data_use, dim_d, quant_name), row.names = F)
write.csv(result_wei, sprintf("result/%s/result_%s_d%d_wei_q%d.csv", 
                              data_use, data_use, dim_d, quant_name), row.names = F)
write.csv(result_max, sprintf("result/%s/result_%s_d%d_max_q%d.csv", 
                              data_use, data_use, dim_d, quant_name), row.names = F)
write.csv(result_gen, sprintf("result/%s/result_%s_d%d_gen_q%d.csv", 
                              data_use, data_use, dim_d, quant_name), row.names = F)




