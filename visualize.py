# =============================================================================
# [모듈 개요]
# 본 파일은 A-Point-Set-Generation 프로젝트의 3D 포인트 클라우드 시각화 유틸리티다.
# 단일 이미지로부터 생성된 3D 포인트 클라우드(11,000개 점의 x,y,z 좌표)를
# matplotlib의 3D scatter plot으로 화면에 그리고 PNG 파일로 저장한다.
#
# 시각화가 필요한 이유:
#   딥러닝 모델이 출력한 포인트 클라우드는 수천 개의 (x,y,z) 숫자 배열이므로
#   텍스트만으로는 형태를 파악하기 어렵다. 이를 3D 산점도(scatter plot)으로
#   그려야 모델이 물류 박스의 3D 형태를 얼마나 잘 복원했는지 직관적으로 확인할 수 있다.
# =============================================================================

# --- import 구간 ---
# 동일한 import가 중복되어 있지만, 원본 코드 로직을 보존하기 위해 그대로 유지한다.
# (중복 import는 Python에서 무시되므로 실행에 영향을 주지 않는다.)

# numpy: 포인트 클라우드는 (N, 3) 형태의 수치 배열이므로 수치 연산을 위해 필요하다.
import numpy as np
# matplotlib.pyplot: 그래프/이미지를 화면에 그리고 파일로 저장하기 위한 핵심 라이브러리다.
import matplotlib.pyplot as plt
# mpl_toolkits.mplot3d.Axes3D: 2D 평면이 아닌 3D 축(x,y,z)을 그리기 위해 필요하다.
# import만 하면 matplotlib의 subplot에 projection='3d' 옵션을 사용할 수 있다.
from mpl_toolkits.mplot3d import Axes3D
# os: 저장 경로의 디렉토리를 생성하거나 파일명을 다룰 때 사용한다.
import os
# 아래 4줄은 위와 동일한 중복 import다. 원본을 보존하기 위해 유지한다.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# torchvision.transforms: PyTorch 텐서 형태의 이미지를 PIL 이미지로 변환하기 위해 사용한다.
# 모델 입력/출력은 텐서지만, matplotlib로 이미지를 출력하려면 PIL 또는 numpy 배열이 필요하다.
import torchvision.transforms as T


# =============================================================================
# [클래스: Visualize]
# 포인트 클라우드를 3D 산점도(scatter plot)로 시각화하는 유틸리티 클래스다.
#
# 이 클래스가 존재하는 이유:
#   학습/평가 과정에서 모델이 생성한 포인트 클라우드가 정답(GT)과 얼마나 유사한지
#   눈으로 확인하려면, 원본 이미지와 3D 점군을 나란히 배치한 시각화 이미지가 필요하다.
#   이 클래스는 그 시각화 결과물을 PNG 파일로 저장하는 역할을 담당한다.
# =============================================================================
class Visualize:
    """
    3D 포인트 클라우드 시각화 클래스.
    단일 포인트 클라우드를 입력받아 원본 이미지와 함께 3D 산점도로 시각화한다.
    """

    def __init__(self, point_cloud):
        """
        시각화할 포인트 클라우드를 인스턴스에 저장한다.
        """
        # point_cloud: (N, 3) 형태의 numpy 배열로, N개 점의 (x, y, z) 좌표를 담는다.
        # 인스턴스 변수로 보관하는 이유: ShowResult에서 재사용하기 위해서다.
        self.point_cloud = point_cloud

    def ShowResult(self, img_tensor, save_path):
        '''
        원본 이미지와 포인트 클라우드를 나란히 시각화하고 저장한다.
        Args:
            img_tensor: 원본 이미지 텐서 (모델 입력으로 사용된 227x227 이미지)
            save_path: 저장할 파일 경로 (확장자 제외, .png가 자동으로 붙는다)
        '''
        # --- 저장 디렉토리 생성 ---
        # save_path의 상위 디렉토리가 존재하지 않을 수 있으므로 미리 생성한다.
        # exist_ok=True: 디렉토리가 이미 있어도 에러를 발생시키지 않기 위해서다.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # --- 이미지 텐서를 PIL 이미지로 변환 ---
        # img_tensor는 (1, C, H, W) 형태(배치 차원 포함)이므로 squeeze(0)로 배치 차원을 제거한다.
        # ToPILImage(): (C, H, W) 텐서를 matplotlib가 출력할 수 있는 PIL 이미지로 변환한다.
        # 텐서를 그대로는 matplotlib로 그릴 수 없기 때문에 이 변환이 필요하다.
        img_np = T.ToPILImage()(img_tensor.squeeze(0))

        # --- figure 생성 ---
        # figsize=(20, 10): 왼쪽에 이미지, 오른쪽에 3D plot을 나란히 배치해야 하므로
        # 가로가 세로의 2배인 넓은 캔버스를 사용한다.
        fig = plt.figure(figsize=(20, 10))

        # --- 왼쪽 subplot: 원본 이미지 ---
        # subplot(121): 1행 2열 중 첫 번째(왼쪽) 위치를 의미한다.
        ax1 = plt.subplot(121)
        # 변환된 PIL 이미지를 출력한다.
        ax1.imshow(img_np)
        # axis('off'): 이미지 주변의 눈금/축을 숨긴다. 이미지 자체만 보여주기 위해서다.
        ax1.axis('off')
        # 어떤 이미지인지 식별할 수 있도록 제목을 단다.
        ax1.set_title('Original Image', fontsize=12)

        # --- 오른쪽 subplot: 3D 포인트 클라우드 ---
        # subplot(122, projection='3d'): 1행 2열 중 두 번째(오른쪽) 위치에 3D 축을 생성한다.
        # projection='3d'가 가능한 것은 상단의 Axes3D import 덕분이다.
        ax2 = plt.subplot(122, projection='3d')

        # --- z값을 색상 매핑용으로 추출 ---
        # 포인트 클라우드의 3번째 열(z 좌표)을 색상 값으로 사용한다.
        # z(깊이)를 색상으로 표현하면, 평면 이미지에서도 점의 앞뒤 깊이를 직관적으로 파악할 수 있다.
        z_values = self.point_cloud[:,2]

        # --- 3D 산점도(scatter) 생성 ---
        # scatter(): 각 점을 (x, y, z) 위치에 작은 원으로 그린다.
        #   c=z_values: 각 점의 색상을 z값에 따라 결정한다 (깊이에 따른 색상 변화).
        #   cmap='viridis': 저색맹(color blindness)에서도 구별이 쉬운 색상 맵을 사용한다.
        #   s=3: 점 크기를 작게 설정한다. 11,000개 점을 그릴 때 겹침을 줄이기 위해서다.
        #   alpha=0.6: 반투명하게 그려, 뒤쪽 점이 앞쪽 점에 가려지지 않도록 한다.
        scatter = ax2.scatter(self.point_cloud[:,0],
                            self.point_cloud[:,1],
                            self.point_cloud[:,2],
                            c=z_values,
                            cmap='viridis',
                            s=3,
                            alpha=0.6)

        # --- 배경 및 그리드 설정 ---
        # 흰색 배경을 사용하는 이유: 점 색상(viridis)과 대비를 높여 가시성을 확보하기 위해서다.
        ax2.set_facecolor('white')
        # 그리드를 얇게 표시하여 3D 공간에서 점의 대략적 위치를 파악할 수 있게 돕는다.
        # alpha=0.3으로 매우 연하게 처리해 점 자체를 가리지 않도록 한다.
        ax2.grid(True, linestyle='-', alpha=0.3)

        # --- 고정된 축 범위 설정 ---
        # 축 범위를 [-30, 30]으로 고정하는 이유:
        #   서로 다른 샘플(이미지)의 포인트 클라우드를 비교할 때, 축 범위가 매번 바뀌면
        #   크기 비교가 어렵기 때문에 모든 시각화에 동일한 스케일을 적용한다.
        x_range = [-30, 30]  # 적절한 값으로 조정 필요
        y_range = [-30, 30]  # 적절한 값으로 조정 필요
        z_range = [-30, 30]  # 적절한 값으로 조정 필요

        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_zlim(z_range)

        # --- 축 비율 고정 ---
        # set_box_aspect([1,1,1]): x/y/z 축의 물리적 길이를 1:1:1로 맞춘다.
        # 이를 하지 않으면 3D 객체가 한 축 방향으로 찌그러져 보여 형태 왜곡이 발생한다.
        ax2.set_box_aspect([1, 1, 1])

        # --- 축 레이블 설정 ---
        # 각 축이 어떤 좌표축인지 표시한다. 포인트 클라우드의 x/y/z 방향을 식별하기 위해서다.
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # --- 시점(카메라 각도) 조정 ---
        # view_init(elev=30, azim=40):
        #   elev(고도각)=30: 위에서 약간 내려다보는 각도. 객체의 윗면이 보이도록 한다.
        #   azim(방위각)=40: 좌우 회전 각도. 객체의 정면이 아닌 비스듬한 각도에서 보여
        #   3차원 깊이감을 더 잘 느낄 수 있게 한다.
        ax2.view_init(elev=30, azim=40)
        ax2.set_title('3D Point Cloud', fontsize=12)

        # --- 전체 figure 배경색 설정 ---
        # subplot 외부의 캔버스 배경도 흰색으로 맞춰, 저장 시 투명/회색 테두리가 생기지 않게 한다.
        fig.patch.set_facecolor('white')

        # --- 전체 타이틀 설정 ---
        # 저장 파일명을 상단에 표시하여, 이미지만 봐도 어떤 샘플인지 식별할 수 있도록 한다.
        plt.suptitle(os.path.basename(save_path), fontsize=14)

        # --- figure 저장 ---
        # bbox_inches='tight': 여백을 최소화하여 이미지가 잘리지 않게 저장한다.
        # dpi=300: 고해상도(인쇄 가능 수준)로 저장하여, 점이 많아도 세밀하게 보이도록 한다.
        # .png 확장자를 명시적으로 붙여 저장한다.
        plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=300)
        # close(fig): 메모리 누수를 방지하기 위해 figure를 명시적으로 닫는다.
        # 루프 안에서 반복 호출될 때 열려있는 figure가 누적되면 메모리 부족이 발생할 수 있다.
        plt.close(fig)

# =============================================================================
# [주석 처리된 이전 버전 Visualize 클래스]
# 아래 블록은 triple-quoted string으로 감싸진 비활성 코드다.
# ShowRandom 메서드를 통해 여러 포인트 클라우드를 무작위로 선택해 시각화하던 이전 구현이다.
# 현재는 ShowResult를 사용하는 위쪽 클래스로 대체되었으나, 참고용으로 원본 그대로 보존한다.
# (코드 로직 변경 없이 주석 블록 자체를 유지한다.)
# =============================================================================
"""
class Visualize:
    def __init__(self, pc_list):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        assert type(pc_list) == list
        assert len(pc_list) > 0
        
        self.pc_list = pc_list
    
    def get_next_visual_dir(self, base_path):
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        
        if not os.path.exists(base_path):
            return base_path
            
        counter = 1
        while True:
            new_path = os.path.join(base_dir, f"{base_name}_{counter}")
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def ShowRandom(self, save_path="/workspace/MODELS/A-Point-Set-Generation-Network-for-3D-Object-Reconstruction-from-a-Single-Image/result/visual"):
        '''
        Plots 6 random images from list of point clouds and saves them to specified path
        Args:
            save_path: Directory path where images should be saved
        '''
        n = 6
        assert n < len(self.pc_list)
        
        save_path = self.get_next_visual_dir(save_path)
        print(f"Saving visualizations to: {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Plot and save each point cloud separately
        for i in range(n):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Calculate z-values for color mapping
            z_values = self.pc_list[i][:,2]
            
            # Create scatter plot with viridis colormap
            scatter = ax.scatter(self.pc_list[i][:,0], 
                               self.pc_list[i][:,1], 
                               self.pc_list[i][:,2],
                               c=z_values,  # Color based on z-coordinate
                               cmap='viridis',  # Use viridis colormap
                               s=3,  # Small point size
                               alpha=0.6)  # Some transparency
            
            # Set white background
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Add grid
            ax.grid(True, linestyle='-', alpha=0.3)
            
            # Set labels and ticks
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Adjust viewing angle
            ax.view_init(elev=30, azim=40)
            
            # Save figure
            save_file = os.path.join(save_path, f'point_cloud_{i}.png')
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
        # Create combined visualization
        fig = plt.figure(figsize=(15, 10))
        for i in range(n):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            
            z_values = self.pc_list[i][:,2]
            scatter = ax.scatter(self.pc_list[i][:,0],
                               self.pc_list[i][:,1],
                               self.pc_list[i][:,2],
                               c=z_values,
                               cmap='viridis',
                               s=2,
                               alpha=0.6)
            
            ax.set_facecolor('white')
            ax.grid(True, linestyle='-', alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=30, azim=40)
        
        fig.patch.set_facecolor('white')
        
        # Save combined figure
        save_file = os.path.join(save_path, 'point_clouds_combined.png')
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
"""
