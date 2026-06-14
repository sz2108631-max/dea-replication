global c "lnage klr lnsize bsize dual debt lnrd indrate own lev"
global d "lnagesq klrsq lnsizesq bsizesq dualsq debtsq lnrdsq indratesq ownsq levsq"

*表1：基准回归
reghdfe res Dea $c ,ab(id year) cl(city)  
outreg2 using jizhun.xls, append dec(4) e(all)
reghdfe res Breadth $c ,ab(id year) cl(city)
outreg2 using jizhun.xls, append dec(4) e(all)
reghdfe res Depth $c ,ab(id year) cl(city)
outreg2 using jizhun.xls, append dec(4) e(all) 

*表2：影响机制检验
*gen Breadth_ic=Breadth*IC
*gen Depth_ic=Depth*IC
reghdfe res Breadth IC Breadth_ic $c ,ab(id year) cl(city)
outreg2 using jizhi.xls, append dec(4) e(all)
reghdfe res Depth IC Depth_ic $c ,ab(id year) cl(city)
outreg2 using jizhi.xls, append dec(4) e(all)

*gen Breadth_cost=Breadth*cost
*gen Depth_cost=Depth*cost
reghdfe res Breadth cost Breadth_cost $c ,ab(id year) cl(city)
outreg2 using jizhi.xls, append dec(4) e(all)
reghdfe res Depth cost Depth_cost $c ,ab(id year) cl(city)
outreg2 using jizhi.xls, append dec(4) e(all)
  
*表3：异质性检验
*数商生态异质性
set seed 54
bdiff, group(shushang_group) model(reghdfe res Dea $c, ab(id year) cl(city)) reps(500) bs first detail
reghdfe res Dea $c if shushang_group==0, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)
reghdfe res Dea $c if shushang_group==1, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)

*行业竞争异质性
set seed 54
bdiff, group(market_group) model(reghdfe res Dea $c, ab(id year) cl(city)) reps(500) bs first detail
reghdfe res Dea $c if market_group==0, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)
reghdfe res Dea $c if market_group==1, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)

*环境不确定性
set seed 54
bdiff, group(huanjing_group) model(reghdfe res Dea $c, ab(id year) cl(city)) reps(500) bs first detail
reghdfe res Dea $c if huanjing_group==0, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)
reghdfe res Dea $c if huanjing_group==1, ab(id year) cl(city)
outreg2 using yizhi.xls, append dec(4) e(all)

*表4：供应链网络位置调节效应回归结果
*gen Dea_PageRankC=Dea*PageRank_C1
*gen Breadth_PageRankC=Breadth*PageRank_C1
*gen Depth_PageRankC=Depth*PageRank_C1
*gen Dea_PageRankP=Dea*PageRank_P1
*gen Breadth_PageRankP=Breadth*PageRank_P1
*gen Depth_PageRankP=Depth*PageRank_P1

reghdfe res Dea PageRank_C1 Dea_PageRankC $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)
reghdfe res Breadth PageRank_C1 Breadth_PageRankC $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)
reghdfe res Depth PageRank_C1 Depth_PageRankC $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)

reghdfe res Dea PageRank_P1 Dea_PageRankP $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)
reghdfe res Breadth PageRank_P1 Breadth_PageRankP $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)
reghdfe res Depth PageRank_P1 Depth_PageRankP $c ,ab(id year) cl(city)
outreg2 using wangluo.xls, append dec(4) e(all)
 
*表5：企业数据要素应用能力与供应链地理距离
reghdfe Disw_s Dea $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)
reghdfe Disw_s Breadth $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)
reghdfe Disw_s Depth $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)

reghdfe Disw_c Dea $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)
reghdfe Disw_c Breadth $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)
reghdfe Disw_c Depth $c ,ab(id year) cl(city)
outreg2 using tuozhan.xls, append dec(4) e(all)

*附表3 数据要素应用能力指标有效性检验：基于数据要素产出角度
reghdfe dig Dea ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe dig Breadth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe dig Depth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)

reghdfe AI Dea ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe AI Breadth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe AI Depth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)

reghdfe learning Dea ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe learning Breadth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)
reghdfe learning Depth ,ab(id year) cl(city)
outreg2 using kouhao.xls, append dec(4) e(all)

*附表4 数据要素应用能力指标有效性检验：基于数据要素产出角度
anova Dea industry_cat
oneway Dea industry_cat, bonferroni

*附表5：描述性统计
sum res Dea Breadth Depth $c

*附表6：内生性检验：工具变量回归
ivreghdfe res (Dea = shandian) $c, ab(id year) cl(city) first r savefirst
ivreghdfe res (Breadth = shandian) $c, ab(id year) cl(city) first r savefirst
ivreghdfe res (Depth = shandian) $c, ab(id year) cl(city) first r savefirst

*附表7：倾向得分匹配
clear
use data
set seed 111
psmatch2 Dea_group ($c), outcome(Dea) kernel n(1) ate logit common
reghdfe res Dea $c [pw=_weight], absorb(id year) cluster(city)

clear
use data
set seed 111
psmatch2 Breadth_group ($c), outcome(Breadth) kernel n(1) ate logit common
reghdfe res Breadth $c [pw=_weight], absorb(id year) cluster(city)

clear
use data
set seed 111
psmatch2 Depth_group ($c), outcome(Depth) kernel n(1) ate logit common
reghdfe res Depth $c [pw=_weight], absorb(id year) cluster(city)

*附表8：稳健性检验：替换变量衡量方式
reghdfe res Dea_count $c ,ab(id year) cl(city)
outreg2 using count.xls, append dec(4) e(all)
reghdfe res Breadth_count $c ,ab(id year) cl(city)
outreg2 using count.xls, append dec(4) e(all)
reghdfe res Depth_count $c ,ab(id year) cl(city) 
outreg2 using count.xls, append dec(4) e(all)

reghdfe res1 Dea $c ,ab(id year) cl(city)
outreg2 using ku.xls, append dec(4) e(all)
reghdfe res1 Breadth $c ,ab(id year) cl(city)
outreg2 using ku.xls, append dec(4) e(all)
reghdfe res1 Depth $c ,ab(id year) cl(city) 
outreg2 using ku.xls, append dec(4) e(all)

*附表9：稳健性检验：更换估计方法
ppmlhdfe res Dea $c, ab(id year) cluster(city)
outreg2 using ppml.xls, append dec(4) e(all) 
ppmlhdfe res Breadth $c, ab(id year) cluster(city)
outreg2 using ppml.xls, append dec(4) e(all)
ppmlhdfe res Depth $c, ab(id year) cluster(city)
outreg2 using ppml.xls, append dec(4) e(all)

*附表10：稳健性检验：调整固定效应
reghdfe res Dea $c ,ab(id year year#ind) cl(city)
outreg2 using guding.xls, append dec(4) e(all) 
reghdfe res Breadth $c ,ab(id year year#ind) cl(city)
outreg2 using guding.xls, append dec(4) e(all) 
reghdfe res Depth $c ,ab(id year year#ind) cl(city)
outreg2 using guding.xls, append dec(4) e(all) 

*附表11：稳健性检验：排除相关政策试点干扰
clear
use data
reghdfe res Dea $c if 全国供应链创新与应用示范城市和示范企业 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)
reghdfe res Breadth $c if 全国供应链创新与应用示范城市和示范企业 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)
reghdfe res Depth $c if 全国供应链创新与应用示范城市和示范企业 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)

reghdfe res Dea $c if 智能建造城市试点 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)
reghdfe res Breadth $c if 智能建造城市试点 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)
reghdfe res Depth $c if 智能建造城市试点 == 0, ab(id year) cl(city)
outreg2 using tichu.xls, append dec(4) e(all)

*附表12：双重机器学习回归 kfolds(5)后续替换为kfolds(5)，单个运行时间大约5小时。
clear
  use data
  global Y res
  global X $c $d i.year i.id
  global D Dea
  *global D Breadth
  *global D Depth
  set seed 42
  ddml init partial, kfolds(5)
  ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv) 
  ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
  ddml crossfit
  ddml estimate, robust

  clear
  use data
  global Y res
  global X $c $d i.year i.id
  global D Dea
  *global D Breadth
  *global D Depth
  set seed 42
  ddml init partial, kfolds(5)
  ddml E[D|X]: pystacked $D $X, type(reg) method(svm) 
  ddml E[Y|X]: pystacked $Y $X, type(reg) method(svm)
  ddml crossfit
  ddml estimate, robust

  clear
  use data
  global Y res
  global X $c $d i.year i.id
  global D Dea
  *global D Breadth
  *global D Depth
  set seed 42
  ddml init partial, kfolds(5)
  ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv) 
  ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
  ddml crossfit
  ddml estimate, robust
  
  clear
  use data
  global Y res
  global X $c $d i.year i.id
  global D Dea
  *global D Breadth
  *global D Depth
  set seed 42
  ddml init partial, kfolds(5)
  ddml E[D|X]: pystacked $D $X, type(reg) method(gradboost) 
  ddml E[Y|X]: pystacked $Y $X, type(reg) method(gradboost)
  ddml crossfit
  ddml estimate, robust
  
*附图567 安慰剂检验
set seed 1234
set more off
local vars Dea Breadth Depth
foreach v of local vars {
    reghdfe res `v' $c, ab(id year) cl(city)
    scalar true_b_`v' = _b[`v']
}
local reps = 5000
foreach v of local vars {
    matrix B_`v' = J(`reps',1,.)
    matrix P_`v' = J(`reps',1,.)
}
forvalues r = 1/`reps' {
    foreach v of local vars {
        tempvar shuffle
        gen `shuffle' = `v'[runiformint(1,_N)]

        quietly reghdfe res `shuffle' $c, ab(id year) cl(city)
        matrix B_`v'[`r',1] = _b[`shuffle']
        quietly test `shuffle' = 0
        matrix P_`v'[`r',1] = r(p)

        drop `shuffle'
    }
    if mod(`r', 100) == 0 {
        di "已完成 `r'/`reps' 次迭代"
    }
}
preserve
    clear
    set obs `reps'
    gen iter = _n
    foreach v of local vars {
        quietly svmat double B_`v', name(b_)   // 生成 b_1
        quietly svmat double P_`v', name(p_)   // 生成 p_1
        rename b_1 placebo_b_`v'
        rename p_1 placebo_p_`v'
    }
    save "placebo_results.dta", replace
restore
append using "placebo_results.dta"
foreach v of local vars {
    quietly count if abs(placebo_b_`v') >= abs(true_b_`v') & !mi(placebo_b_`v')
    scalar emp_p_`v' = r(N)/`reps'

    di _n "变量 `v'  真实系数: " true_b_`v' "  经验 p 值: " emp_p_`v'
}

use placebo_results,clear
twoway (scatter placebo_p_Dea placebo_b_Dea, msymbol(smcircle_hollow) mcolor(blue)) (kdensity placebo_b_Dea, yaxis(2)), title("安慰剂检验") xlabel(-0.0025(0.001)0.0025 0) xline(0.0016, lcolor(red) lp(dash)) xtitle("估计系数")  yline(0.1, lp(shortdash))ytitle("P值", axis(1)) ytitle("核密度", axis(2)) legend(label(1 "P值") label(2 "核密度")) graphregion(color(white))
 
twoway (scatter placebo_p_Breadth placebo_b_Breadth, msymbol(smcircle_hollow) mcolor(blue)) (kdensity placebo_b_Breadth, yaxis(2)), title("安慰剂检验") xlabel(-0.0006(0.0002)0.0006 0) xline(0.0004, lcolor(red) lp(dash)) xtitle("估计系数")  yline(0.1, lp(shortdash))ytitle("P值", axis(1)) ytitle("核密度", axis(2)) legend(label(1 "P值") label(2 "核密度")) graphregion(color(white))

twoway (scatter placebo_p_Depth placebo_b_Depth, msymbol(smcircle_hollow) mcolor(blue)) (kdensity placebo_b_Depth, yaxis(2)), title("安慰剂检验") xlabel(-0.003(0.001)0.003 0) xline(0.0018, lcolor(red) lp(dash)) xtitle("估计系数")  yline(0.1, lp(shortdash))ytitle("P值", axis(1)) ytitle("核密度", axis(2)) legend(label(1 "P值") label(2 "核密度")) graphregion(color(white))

tabstat placebo_b_Dea placebo_p_Dea placebo_b_Breadth placebo_p_Breadth placebo_b_Depth placebo_p_Depth,stat(count mean p5 p10 p25 median p75 p90 p95 sd) col(stat)