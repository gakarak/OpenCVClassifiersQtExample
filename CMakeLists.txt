cmake_minimum_required(VERSION 2.6)

#
# проверка Unicode (UTF-8):
# 偉大和強大的俄語。
# اللغة الروسية العظمى والأقوياء.
# Η μεγάλη και πανίσχυρη ρωσική γλώσσα.
# השפה הרוסיה הגדולה והעצומה.
# 大いなるロシア語。
# 위대하고 강력한 러시아어 언어입니다.
# Wielki i potężny język rosyjski.
#

PROJECT( ExampleOpenCVClassifiersQt )

set( TARGET_0 ExampleOpenCVClassifiersQt )

if(NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

FIND_PACKAGE( OpenCV REQUIRED )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)

include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5Core_INCLUDE_DIRS})

add_definitions(${Qt5Widgets_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

##INCLUDE(${QT_USE_FILE})
##INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
##ADD_DEFINITIONS(${QT_DEFINITIONS})


SET(PRJ_SRC	classifier_ann.cpp
		classifier_bt.cpp
		classifier_dt.cpp
		classifier_emc.cpp
		classifier_em.cpp
		classifier_ert.cpp
		classifier_gbt.cpp
		classifierinterface.cpp
		classifier_knn.cpp
		classifier_nbc.cpp
		classifier_rf.cpp
		classifier_svm.cpp
		drawlabel.cpp
		mainwindow.cpp
		main.cpp)

SET(PRJ_HDR	classifier_ann.h
		classifier_bt.h
		classifier_dt.h
		classifier_emc.h
		classifier_em.h
		classifier_ert.h
		classifier_gbt.h
		classifierinterface.h
		classifier_knn.h
		classifier_nbc.h
		classifier_rf.h
		classifier_svm.h
		drawlabel.h
		mainwindow.h)


SET(PRJ_FORMS	mainwindow.ui )
SET(PRJ_RES	imgres.qrc    )

##QT4_WRAP_CPP(PRJ_HDR_MOC	${PRJ_HDR})
##QT4_WRAP_UI(PRJ_FORMS_HDR	${PRJ_FORMS})
##QT4_ADD_RESOURCES(PRJ_RES_RCC	${PRJ_RES})

QT5_WRAP_UI(PRJ_FORMS_HDR	${PRJ_FORMS})
QT5_ADD_RESOURCES(PRJ_RES_RCC	${PRJ_RES})


ADD_EXECUTABLE( ${TARGET_0}_${CMAKE_BUILD_TYPE}		${PRJ_SRC} ${PRJ_HDR_MOC} ${PRJ_FORMS_HDR} ${PRJ_RES_RCC})
TARGET_LINK_LIBRARIES(	${TARGET_0}_${CMAKE_BUILD_TYPE}
			${Qt5Widgets_LIBRARIES}
			${Qt5Core_LIBRARIES}
			${OpenCV_LIBS})


##get_cmake_property(_variableNames VARIABLES)
##foreach (_variableName ${_variableNames})
##   message(STATUS "${_variableName}=${${_variableName}}")
##endforeach()
