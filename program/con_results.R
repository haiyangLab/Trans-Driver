library(ggplot2)
library(patchwork)

databar<-read.table('C:/Users/zhanglei/Desktop/论文材料/results_3/multi_vs_monoomics.txt', header=T, sep='\t', encoding="UTF-8")
print(databar)
#q1<-ggplot(data=databar, mapping=aes(x =Methods,  y = Fisher_value, fill=Dataset))+
#  geom_bar(stat="identity",position=position_dodge(0.75))+labs(title = "Fisher exact results in different methods")+theme(plot.title = element_text(hjust = 0.5)) 
#q1

#databar1<-read.table('C:/Users/zhanglei/Desktop/11_6.txt', header=T, sep='\t', encoding="UTF-8")
#print(databar1)
#q2<-ggplot(data=databar1, mapping=aes(x = Cancer, y = Importance,fill=Features))+
#geom_bar(stat="identity",position=position_dodge(0.75))+labs(title = "Features importance")+theme(plot.title = element_text(hjust = 0.5))  
#q2
#theme函数设置
theme_bar1 <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),#去网格线
          #axis.title.x = element_blank(),#去x轴标签
          axis.title.y=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title.x=element_text(face = "bold",size = 12),#y轴标签加粗及字体大小
          axis.title=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.text = element_text(face = "bold",size = 12),#坐标轴刻度标签加粗
          #axis.text.x = element_text(angle=45, hjust = 0.5, vjust = 0.5, face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),#坐标轴刻度线
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),#去除图例标题
          # legend.justification=c(1,0),#图例在画布的位置(绘图区域外)
          #legend.position=c(0.5, 0.7),#图例在绘图区域的位置
          legend.position='none',#图例放在顶部
          legend.direction = "vertical",#设置图例水平放置
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 12,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)#图例与绘图区域边缘的距离
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}
sorted1_list <- c('CGC fisher pancan', 'TCGA fisher pancan', 'TCGA', 'CGC', 'Fisher each cancer')
#绘图
#databar$x <- factor(databar$x,levels=c("Trans-Driver","2020plus","MuSiC","CHASM","VEST", "ActiveDriver", "e-Driver", "CompositeDriver"))
q1<-ggplot(data=databar, mapping=aes(x=factor(evaluate, level=sorted1_list),  y=value, fill=data))+
  geom_bar(stat="identity",position=position_dodge(0.75), width=0.6)+
  coord_cartesian(ylim=c(0,160))+
  labs(x="Evaluation",y = "Value",title = "Multi-omics and mutation data comparison")+
  scale_y_continuous(expand = c(0, 0))+#消除x轴与绘图区的间隙
  scale_fill_manual(values =c("#CC0000", "#006600", "#669999", "#00CCCC"))+
  theme_bar1()



databar1<-read.table('C:/Users/zhanglei/Desktop/论文材料/results_3/pancan_fea_imp_data.txt', header=T, sep='\t', encoding="UTF-8")
print(databar1)
theme_bar2 <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),#去网格线
          #axis.title.x = element_blank(),#去x轴标签
          axis.title.y=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title.x=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.text = element_text(face = "bold",size = 12),#坐标轴刻度标签加粗
          
          # axis.ticks = element_line(color='black'),#坐标轴刻度线
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),#去除图例标题
          # legend.justification=c(1,0),#图例在画布的位置(绘图区域外)
          #legend.position='none',#图例在绘图区域的位置
          legend.position='none',#图例放在顶部
          legend.direction = "horizontal",#设置图例水平放置
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 12,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)#图例与绘图区域边缘的距离
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}

sorted2_list <- c('mutations', 'other', 'cnv', 'exp','dna_meth')
q2<-ggplot(data=databar1, mapping=aes(x=factor(Features	, level=sorted2_list), y = Importance,fill=Cancer))+
  geom_bar(stat="identity",position=position_dodge(0.1), width=0.4)+
  coord_cartesian(ylim=c(0, 0.6))+
  scale_y_continuous(expand = c(0, 0))+#消除x轴与绘图区的间隙
  labs(x="Features",y = "Importance",title = "Pan cacner features importances")+
  scale_fill_manual(values =c("#CC0000", "#006600", "#669999", "#00CCCC"))+
  theme_bar2()

databar3<-read.table('C:/Users/zhanglei/Desktop/论文材料/results_3/each_cancer_fea_3_4.txt', header=T, sep='\t', encoding="UTF-8")
print(databar3)
theme_bar3 <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          plot.title = element_text(hjust = 0.5, face = "bold",size = 14),
          panel.border=element_rect(fill='transparent', color='transparent'),
          panel.grid=element_blank(),#去网格线
          #axis.title.x = element_blank(),#去x轴标签
          axis.title.y=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title.x=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.title=element_text(face = "bold",size = 14),#y轴标签加粗及字体大小
          axis.text = element_text(face = "bold",size = 12),#坐标轴刻度标签加粗
          axis.text.x = element_text(angle=45, hjust = 0.5, vjust = 0.5, face = "bold",size = 12),
          # axis.ticks = element_line(color='black'),#坐标轴刻度线
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),#去除图例标题
          # legend.justification=c(1,0),#图例在画布的位置(绘图区域外)
          #legend.position=c(0.5, 0.7),#图例在绘图区域的位置
          legend.position='none',#图例放在顶部
          legend.direction = "vertical",#设置图例水平放置
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",size = 12,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          # legend.margin=margin(0,0,-7,0)#图例与绘图区域边缘的距离
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}
q3<-ggplot(data=databar3, mapping=aes(x = Cancer, y = Importance,fill=Features))+
  geom_bar(stat="identity",position=position_dodge(0.75))+
  labs(title = "Features importance")+
  theme_bar3()+ 
  coord_cartesian(ylim=c(0,0.75))+
  scale_y_continuous(expand = c(0, 0))#消除x轴与绘图区的间隙



p1 <- q1 + ggtitle('A')
p2 <- q2 + ggtitle('B')
p3 <- q3 + ggtitle('C')

p <- (p1)/(p2+p3)
p

