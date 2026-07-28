;; Xiao : 新增文件，用于本仓库对上游的扩展。
(define (stream pr2-pick-place)
  (:stream sample-grasp
    :inputs (?o)
    :domain (IsObject ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream sample-place-pose
    :inputs (?o ?s)
    :domain (and (IsObject ?o) (Surface ?s))
    :outputs (?p)
    :certified (and (Pose ?p) (Supported ?o ?s ?p))
  )

  (:stream inverse-kinematics-pick
    :inputs (?r ?o ?g ?p)
    :domain (and (Robot ?r) (IsObject ?o) (Grasp ?o ?g) (Pose ?p))
    :outputs (?bq ?aq)
    :certified (and (BConf ?bq) (AConf ?aq) (KinPick ?r ?o ?g ?bq ?aq ?p))
  )

  (:stream inverse-kinematics-place
    :inputs (?r ?o ?g ?p)
    :domain (and (Robot ?r) (IsObject ?o) (Grasp ?o ?g) (Pose ?p))
    :outputs (?bq ?aq)
    :certified (and (BConf ?bq) (AConf ?aq) (KinPlace ?r ?o ?g ?bq ?aq ?p))
  )

  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :outputs (?path)
    :certified (and (Motion ?path) (BaseMotion ?q1 ?q2 ?path))
  )
)
