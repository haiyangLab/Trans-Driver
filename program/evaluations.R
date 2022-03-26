library(ggplot2)

databar<-read.table('../results/pancan_fisher.txt', header=T, sep='\t', encoding="UTF-8")
print(databar)
theme_bar <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),
          #axis.title.x = element_blank(),
          axis.title.y=element_text(face = "bold",size = 14),
          axis.title.x=element_text(face = "bold",size = 14),
          axis.title=element_text(face = "bold",size = 14),
          axis.text = element_text(face = "bold",size = 14),
          # axis.ticks = element_line(color='black'),
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),
          # legend.justification=c(1,0),
          #legend.position=c(0.28, 0.9),
          legend.position='top',
          legend.direction = "horizontal",
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 8,margin = margin(r=2)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}

q1<-ggplot(data=databar, mapping=aes(x=factor(Dataset),  y=Fisher_value, fill=Methods))+
  geom_bar(stat="identity",position=position_dodge(0.75), width=0.6)+
  coord_cartesian(ylim=c(0,160))+
  labs(x="Data set",y = "-log10 P-value",title = "Fisher exact results in different methods")+
  scale_y_continuous(expand = c(0, 0))+
  scale_fill_manual(values =c("#CC0000", "#006600", "#669999", "#00CCCC", "#660099", "#CC0066", "#FF9999", "#FF9900"))+
  theme_bar()

databar2<-read.table('../results/differ_methods_con.txt', header=T, sep='\t', encoding="UTF-8")
print(databar2)
theme_bar2 <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),
          #axis.title.x = element_blank(),
          axis.title.y=element_text(face = "bold",size = 14),
          axis.title.x=element_text(face = "bold",size = 14),
          axis.title=element_text(face = "bold",size = 14),
          axis.text = element_text(face = "bold",size = 12),
          axis.text.x = element_text(angle=45, hjust = 0.5, vjust = 0.5, face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),
          #legend.justification=c(1,0),
          #legend.position=c(0.75, 0.9),
          legend.position='none',
          legend.direction = "horizontal",
          #legend.direction = "vertical",
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 8,margin = margin(r=2)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}

sorted1_list <- c('CompositeDriver', 'Trans-Driver', '2020plus', 'ActiveDriver',  'e-Driver', 'OncodriveCLUST', 'MuSiC', 'CHASM')
q2<-ggplot(data=databar2, mapping=aes(x = factor(Methods, level = sorted1_list), y = Value, fill=Con))+
  geom_bar(stat="identity",position="stack",width = 0.9)+
  labs(x="Methods",y = "Fraction of predicted drivers")+
  theme_bar2()+ 
  scale_fill_brewer(palette = "Spectral") +
  coord_cartesian(ylim=c(0,1))+
  scale_y_continuous(expand = c(0, 0))

theme_bar3 <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),
          #axis.title.x = element_blank(),
          axis.title.y=element_text(face = "bold",size = 14),
          axis.title.x=element_text(face = "bold",size = 14),
          axis.title=element_text(face = "bold",size = 14),
          axis.text = element_text(face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),
          # legend.justification=c(1,0),
          #legend.position=c(0.85, 0.4),
          legend.position='none',
          #legend.direction = "horizontal",
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 8,margin = margin(r=2)),
          legend.background = element_rect( linetype="solid",colour ="black"),
          #legend.margin=margin(0,0,-7,0),
          #legend.box.margin =margin(-10,0,0,0)
    )
}

sorted2_list <- c('ActiveDriver', 'e-Driver','CHASM',  'CompositeDriver', 'OncodriveCLUST' ,'MuSiC', '2020plus', 'Trans-Driver')

csdn_box<-read.table('../results/each_cancer_fisher_result.txt', header=T, sep='\t', encoding="UTF-8")
q3 <- ggplot(csdn_box, aes(x = factor(Methods, level = sorted2_list), y = P_values,  ))+ 
  geom_boxplot(aes(fill = Methods),position=position_dodge(0.1),width=0.6)+ 
  labs(x="Methods",y="-log10 P-values",title = "Each cancer fisher exact results in different methods")+
  scale_fill_manual(values = c("#CC0000", "#006600", "#669999", "#00CCCC", "#660099", "#DD1166", "#FF9999", "#FF9900", "#FFA500"))+
  theme_bar3() +
  #scale_y_continuous(expand = c(0, 0))+
  coord_flip()

q1 <- q1 + ggtitle('A')
q2 <- q2 + ggtitle('B')
q3 <- q3 + ggtitle('C')

q <- (q1) / (q2 + q3)
q
